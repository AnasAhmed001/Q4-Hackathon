from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict
import time
import asyncio


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to limit requests to 10 req/hour anonymous, 50 req/hour authenticated
    """
    def __init__(self, app, anonymous_limit: int = 10, authenticated_limit: int = 50, window: int = 3600):
        super().__init__(app)
        self.anonymous_limit = anonymous_limit
        self.authenticated_limit = authenticated_limit
        self.window = window  # 1 hour in seconds
        self.requests = defaultdict(list)  # Store request timestamps by identifier

    def get_identifier(self, request: Request) -> str:
        """
        Get identifier for rate limiting (user ID if authenticated, IP address if not)
        """
        # Check if user is authenticated (simplified - in real app, you'd check JWT, session, etc.)
        user_id = request.headers.get("user-id")  # This would be from auth
        if user_id:
            return f"user:{user_id}"
        else:
            # Use IP address for anonymous users
            client_ip = request.client.host
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            return f"ip:{client_ip}"

    def is_rate_limited(self, identifier: str, is_authenticated: bool) -> bool:
        """
        Check if the identifier is rate limited
        """
        current_time = time.time()
        limit = self.authenticated_limit if is_authenticated else self.anonymous_limit

        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window
        ]

        # Check if limit is exceeded
        if len(self.requests[identifier]) >= limit:
            return True

        # Add current request
        self.requests[identifier].append(current_time)
        return False

    async def dispatch(self, request: Request, call_next):
        # Determine if user is authenticated
        user_id = request.headers.get("user-id")
        is_authenticated = user_id is not None

        # Get identifier and check rate limit
        identifier = self.get_identifier(request)
        if self.is_rate_limited(identifier, is_authenticated):
            if is_authenticated:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.authenticated_limit} requests per hour for authenticated users"
                )
            else:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.anonymous_limit} requests per hour for anonymous users"
                )

        response = await call_next(request)
        return response