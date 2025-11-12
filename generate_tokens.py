#!/usr/bin/env python3
"""
Standalone token generator for Nordlys Fetcher.
Run this script to generate time-limited access tokens for your clients.

Usage:
    python generate_tokens.py                    # Interactive mode
    python generate_tokens.py client1 72         # Generate token for client1, valid 72h
    python generate_tokens.py test 0.00833       # Generate test token (30 seconds)
"""

import os
import sys
import hmac
import hashlib
import base64
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-me-in-production")
DEFAULT_EXPIRY_HOURS = float(os.getenv("TOKEN_EXPIRY_HOURS", str(72)))


def generate_token(client_id: str, expiry_hours: float) -> str:
    """Generate a time-limited token."""
    issued_at = int(time.time())
    payload = f"{client_id}:{issued_at}:{expiry_hours}"
    signature = hmac.new(
        TOKEN_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()[:16]
    token_data = f"{payload}:{signature}"
    return base64.urlsafe_b64encode(token_data.encode()).decode()


def verify_token(token: str) -> dict:
    """Verify and decode a token."""
    try:
        token_data = base64.urlsafe_b64decode(token.encode()).decode()
        parts = token_data.split(':')
        
        if len(parts) != 4:
            return {"valid": False, "error": "Invalid format"}
        
        client_id, issued_at_str, expiry_hours_str, signature = parts
        payload = f"{client_id}:{issued_at_str}:{expiry_hours_str}"
        
        expected_signature = hmac.new(
            TOKEN_SECRET.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        
        if not hmac.compare_digest(signature, expected_signature):
            return {"valid": False, "error": "Invalid signature"}
        
        issued_at = int(issued_at_str)
        expiry_hours = float(expiry_hours_str)
        issued_dt = datetime.fromtimestamp(issued_at)
        expires_at = issued_dt + timedelta(hours=expiry_hours)
        now = datetime.now()
        
        if now > expires_at:
            return {"valid": False, "error": "Expired", "expired_at": expires_at}
        
        remaining = (expires_at - now).total_seconds()
        
        return {
            "valid": True,
            "client_id": client_id,
            "issued_at": issued_dt,
            "expires_at": expires_at,
            "remaining_seconds": int(remaining)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def format_time(seconds: float) -> str:
    """Format seconds to human-readable."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{seconds/3600:.1f}h"


def interactive_mode():
    """Interactive token generation."""
    print("=" * 60)
    print("🔐  NORDLYS FETCHER - TOKEN GENERATOR")
    print("=" * 60)
    print(f"Token Secret: {TOKEN_SECRET[:10]}..." if len(TOKEN_SECRET) > 10 else "⚠️  DEFAULT SECRET")
    print(f"Default Expiry: {DEFAULT_EXPIRY_HOURS} hours")
    print("=" * 60)
    print()
    
    while True:
        print("\n📋 OPTIONS:")
        print(f"  1. Generate token ({DEFAULT_EXPIRY_HOURS:g} hours)")
        print("  2. Generate token (custom expiry)")
        print("  3. Generate TEST token (30 seconds)")
        print("  4. Verify existing token")
        print("  5. Batch generate tokens")
        print("  6. Exit")
        print()
        
        choice = input("Choose option [1-6]: ").strip()
        
        if choice == '1':
            client_id = input("\nClient ID: ").strip()
            if not client_id:
                print("❌ Client ID required")
                continue
            
            token = generate_token(client_id, DEFAULT_EXPIRY_HOURS)
            expires = datetime.now() + timedelta(hours=DEFAULT_EXPIRY_HOURS)
            
            print("\n" + "=" * 60)
            print(f"✅ TOKEN GENERATED")
            print("=" * 60)
            print(f"Client ID:  {client_id}")
            print(f"Valid for:  {DEFAULT_EXPIRY_HOURS} hours")
            print(f"Expires:    {expires.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print(f"Token:")
            print(f"  {token}")
            print()
            print(f"Access URL:")
            print(f"  http://your-domain.com/?token={token}")
            print("=" * 60)
        
        elif choice == '2':
            client_id = input("\nClient ID: ").strip()
            expiry_str = input("Expiry (hours): ").strip()
            
            if not client_id:
                print("❌ Client ID required")
                continue
            
            try:
                expiry = float(expiry_str)
                if expiry <= 0:
                    print("❌ Expiry must be positive")
                    continue
                
                token = generate_token(client_id, expiry)
                expires = datetime.now() + timedelta(hours=expiry)
                
                print("\n" + "=" * 60)
                print(f"✅ CUSTOM TOKEN GENERATED")
                print("=" * 60)
                print(f"Client ID:  {client_id}")
                print(f"Valid for:  {expiry} hours")
                print(f"Expires:    {expires.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                print(f"Token:")
                print(f"  {token}")
                print("=" * 60)
            except ValueError:
                print("❌ Invalid expiry value")
        
        elif choice == '3':
            client_id = input("\nClient ID (test): ").strip() or "test"
            
            token = generate_token(client_id, 30/3600)  # 30 seconds
            expires = datetime.now() + timedelta(seconds=30)
            
            print("\n" + "=" * 60)
            print(f"⚠️  TEST TOKEN (30 SECONDS ONLY)")
            print("=" * 60)
            print(f"Client ID:  {client_id}")
            print(f"Valid for:  30 seconds")
            print(f"Expires:    {expires.strftime('%H:%M:%S')}")
            print()
            print(f"Token:")
            print(f"  {token}")
            print()
            print(f"⏰ Quick! Test it now:")
            print(f"  http://localhost:8000/?token={token}")
            print("=" * 60)
        
        elif choice == '4':
            token = input("\nToken to verify: ").strip()
            
            if not token:
                print("❌ Token required")
                continue
            
            result = verify_token(token)
            
            print("\n" + "=" * 60)
            if result['valid']:
                print("✅ TOKEN IS VALID")
                print("=" * 60)
                print(f"Client ID:   {result['client_id']}")
                print(f"Issued:      {result['issued_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Expires:     {result['expires_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Remaining:   {format_time(result['remaining_seconds'])}")
            else:
                print("❌ TOKEN IS INVALID")
                print("=" * 60)
                print(f"Error: {result['error']}")
            print("=" * 60)
        
        elif choice == '5':
            count_str = input("\nNumber of tokens to generate: ").strip()
            prefix = input("Client ID prefix (e.g., 'client'): ").strip() or "client"
            expiry_str = input(f"Expiry hours (default {DEFAULT_EXPIRY_HOURS}): ").strip()
            
            try:
                count = int(count_str)
                expiry = float(expiry_str) if expiry_str else DEFAULT_EXPIRY_HOURS
                
                print("\n" + "=" * 60)
                print(f"BATCH GENERATION - {count} TOKENS")
                print("=" * 60)
                
                for i in range(1, count + 1):
                    client_id = f"{prefix}{i}"
                    token = generate_token(client_id, expiry)
                    expires = datetime.now() + timedelta(hours=expiry)
                    
                    print(f"\n{i}. Client: {client_id}")
                    print(f"   Token: {token}")
                    print(f"   Expires: {expires.strftime('%Y-%m-%d %H:%M')}")
                
                print("\n" + "=" * 60)
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


def cli_mode(client_id: str, expiry_hours: float):
    """CLI mode for quick token generation."""
    token = generate_token(client_id, expiry_hours)
    expires = datetime.now() + timedelta(hours=expiry_hours)
    
    print(f"Client: {client_id}")
    print(f"Expiry: {expiry_hours}h ({expires.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Token:  {token}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # CLI mode: python generate_tokens.py client1 72
        try:
            client_id = sys.argv[1]
            expiry = float(sys.argv[2])
            cli_mode(client_id, expiry)
        except ValueError:
            print("Usage: python generate_tokens.py <client_id> <expiry_hours>")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()