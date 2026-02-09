"""Shark Robot Controller for Cat Litter Monitor.

This module provides a working Shark vacuum controller using the reverse-engineered
stakra API.

To use:
1. Create a .shark_token file in config/ with your Auth0 refresh token:
   echo '{"auth0_refresh_token": "YOUR_TOKEN"}' > config/.shark_token

2. Update settings.yaml with your device credentials:
   robot:
     enabled: true
     room_name: "Litter"
     type: "shark"
     household_id: "YOUR_HOUSEHOLD_ID"
     dsn: "YOUR_DSN"
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import time

import aiohttp

from src.robot.interface import BaseRobotController

logger = logging.getLogger(__name__)

# Stakra API constants (from old project reverse engineering)
AUTH0_URL = "https://login.sharkninja.com"
AUTH0_CLIENT_ID = "YOUR_AUTH0_CLIENT_ID"
STAKRA_HOST = "stakra.slatra.thor.skegox.com"
STAKRA_BASE = f"https://{STAKRA_HOST}"
STAKRA_API_KEY = "YOUR_STAKRA_API_KEY"

# Token file path
TOKEN_FILE = Path(__file__).resolve().parent.parent.parent / "config" / ".shark_token"


class OperatingModes:
    STOP = 0
    START = 2
    PAUSE = 3
    RETURN = 4


def _sign_stakra_request(method: str, path: str, query: str = "", body: bytes = b"") -> dict:
    """Generate HMAC-signed headers for stakra API request."""
    sn_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sn_nonce = secrets.token_hex(16)
    body_hash = hashlib.sha256(body).hexdigest()
    
    # Canonical request
    cr = "\n".join([
        method, path, query,
        f"host:{STAKRA_HOST}",
        f"x-sn-date:{sn_date}",
        f"x-sn-nonce:{sn_nonce}",
        "",
        "host;x-sn-date;x-sn-nonce",
        body_hash,
    ])
    cr_hash = hashlib.sha256(cr.encode()).hexdigest()
    
    # String to sign
    scope = f"{sn_date}/*/end-user-api/sn_request"
    sts = f"SN-HMAC-SHA256\n{sn_date}\n{scope}\n{cr_hash}"
    
    # Key derivation
    date_only = sn_date[:8]
    k_date = hmac.new(("SN" + STAKRA_API_KEY).encode(), date_only.encode(), hashlib.sha256).digest()
    k_region = hmac.new(k_date, b"*", hashlib.sha256).digest()
    k_service = hmac.new(k_region, b"end-user-api", hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"sn_request", hashlib.sha256).digest()
    
    signature = hmac.new(k_signing, sts.encode(), hashlib.sha256).hexdigest()
    
    return {
        "x-sn-date": sn_date,
        "x-sn-nonce": sn_nonce,
        "x-iotn-request-signature": (
            f"SN-HMAC-SHA256 Credential={AUTH0_CLIENT_ID}/{scope}, "
            f"SignedHeaders=host;x-sn-date;x-sn-nonce, Signature={signature}"
        ),
    }


def _save_token(token: str):
    """Persist Auth0 refresh token to disk."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"auth0_refresh_token": token}))
    logger.info(f"Auth0 refresh token saved to {TOKEN_FILE}")


def _load_token() -> str | None:
    """Load persisted Auth0 refresh token."""
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            return data.get("auth0_refresh_token")
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
    return None


class StakraClient:
    """Async client for SharkNinja stakra API."""
    
    def __init__(self, auth0_refresh_token: str, household_id: str, dsn: str):
        self.auth0_refresh_token = auth0_refresh_token
        self.household_id = household_id
        self.dsn = dsn
        self._id_token: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._token_expires_at: float = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    @property
    def token_expiring_soon(self) -> bool:
        return time.time() > (self._token_expires_at - 3600)
    
    async def authenticate(self):
        """Refresh Auth0 token."""
        session = await self._get_session()
        
        resp = await session.post(f"{AUTH0_URL}/oauth/token", json={
            "grant_type": "refresh_token",
            "client_id": AUTH0_CLIENT_ID,
            "refresh_token": self.auth0_refresh_token,
        })
        
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Auth0 refresh failed ({resp.status}): {text}")
        
        data = await resp.json()
        self._id_token = data["id_token"]
        self.auth0_refresh_token = data["refresh_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 86400)
        
        _save_token(self.auth0_refresh_token)
        logger.info("Auth0 refresh successful")
    
    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self._id_token}",
            "content-type": "application/json",
            "accept": "*/*",
            "x-api-key": STAKRA_API_KEY,
            "x-iotn-caller": "ENDUSER_MOBILEAPP",
            "user-agent": "snjs",
        }
    
    async def _request(self, method: str, path: str, query: str = "", body: bytes = b""):
        """Make signed request to stakra API."""
        session = await self._get_session()
        url = f"{STAKRA_BASE}{path}"
        if query:
            url += f"?{query}"
        
        headers = self._base_headers()
        headers.update(_sign_stakra_request(method, path, query, body))
        
        kwargs = {"headers": headers}
        if body:
            kwargs["data"] = body
        
        if method == "GET":
            resp = await session.get(url, **kwargs)
        elif method == "PATCH":
            resp = await session.patch(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return resp
    
    async def get_device(self) -> dict:
        """Get device info including shadow."""
        path = f"/devicesEndUserController/{self.household_id}/devices/{self.dsn}"
        resp = await self._request("GET", path)
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Get device failed: {text}")
        return await resp.json()
    
    async def set_desired_properties(self, properties: dict):
        """Set desired properties on device shadow."""
        path = f"/devicesEndUserController/{self.household_id}/devices/{self.dsn}"
        body = json.dumps({
            "shadow": {
                "properties": {
                    "desired": properties
                }
            }
        }).encode()
        
        resp = await self._request("PATCH", path, body=body)
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Set properties failed: {text}")
        return await resp.json()
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class SharkController(BaseRobotController):
    """Shark vacuum controller using stakra API.
    
    This is the working implementation from the old project.
    """
    
    def __init__(
        self,
        household_id: str,
        dsn: str,
        auth0_refresh_token: str | None = None,
        floor_id: str | None = None,
    ):
        self.household_id = household_id
        self.dsn = dsn
        self._auth0_refresh_token = auth0_refresh_token or _load_token()
        self._floor_id = floor_id
        self._room_list: list[str] = []
        
        self._client: StakraClient | None = None
        self._connected = False
        self._state = "unknown"
    
    async def connect(self) -> bool:
        if not self._auth0_refresh_token:
            logger.error("No Auth0 refresh token available")
            return False
        
        try:
            self._client = StakraClient(
                auth0_refresh_token=self._auth0_refresh_token,
                household_id=self.household_id,
                dsn=self.dsn,
            )
            await self._client.authenticate()
            
            # Get device info
            device = await self._client.get_device()
            reported = device.get("shadow", {}).get("properties", {}).get("reported", {})
            
            # Extract room list
            room_list_str = reported.get("Robot_Room_List", {}).get("value", "")
            if room_list_str and ":" in room_list_str:
                parts = room_list_str.split(":")
                if not self._floor_id:
                    self._floor_id = parts[0]
                self._room_list = parts[1:]
                logger.info(f"Floor {self._floor_id} rooms: {self._room_list}")
            
            self._connected = True
            logger.info("Shark controller connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Shark: {e}")
            return False
    
    async def dispatch(self, room: str | None = None) -> bool:
        if not self._client:
            return False
        
        try:
            # Check if paused and stop first
            device = await self._client.get_device()
            reported = device.get("shadow", {}).get("properties", {}).get("reported", {})
            op_mode = reported.get("Operating_Mode", {}).get("value")
            
            if op_mode == OperatingModes.PAUSE:
                logger.info("Robot paused, stopping first...")
                await self._client.set_desired_properties({
                    "Operating_Mode": OperatingModes.STOP
                })
                await asyncio.sleep(2)
            
            props = {"Operating_Mode": OperatingModes.START}
            
            # Room-specific cleaning
            if room and self._floor_id and room in self._room_list:
                areas_payload = json.dumps({
                    "areas_to_clean": {"UserRoom": [room]},
                    "clean_count": 1,
                    "floor_id": self._floor_id,
                    "cleantype": "dry",
                })
                props["AreasToClean_V3"] = areas_payload
                logger.info(f"Dispatching to room: {room}")
            else:
                logger.info("Dispatching for full house clean")
            
            await self._client.set_desired_properties(props)
            self._state = "cleaning"
            return True
            
        except Exception as e:
            logger.error(f"Dispatch failed: {e}")
            return False
    
    async def stop(self) -> bool:
        if not self._client:
            return False
        
        try:
            await self._client.set_desired_properties({
                "Operating_Mode": OperatingModes.STOP
            })
            self._state = "stopped"
            logger.info("Robot stopped")
            return True
        except Exception as e:
            logger.error(f"Stop failed: {e}")
            return False
    
    async def return_to_dock(self) -> bool:
        if not self._client:
            return False
        
        try:
            await self._client.set_desired_properties({
                "Operating_Mode": OperatingModes.RETURN
            })
            self._state = "returning"
            logger.info("Robot returning to dock")
            return True
        except Exception as e:
            logger.error(f"Return failed: {e}")
            return False
    
    def is_available(self) -> bool:
        # Simplified - would check actual robot state in full implementation
        return self._connected and self._state != "cleaning"


# Register this controller
def create_shark_controller(**kwargs) -> SharkController:
    """Factory function for Shark controller."""
    return SharkController(
        household_id=kwargs.get("household_id", ""),
        dsn=kwargs.get("dsn", ""),
        auth0_refresh_token=kwargs.get("auth0_refresh_token"),
        floor_id=kwargs.get("floor_id"),
    )
