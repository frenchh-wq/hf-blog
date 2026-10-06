"""monitoring: Telegram / Discord alerts (no-ops when not configured)."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

log = logging.getLogger("agent.notify")


class Notifier:
    def __init__(self, telegram_token: str = "", telegram_chat_id: str = "", discord_webhook: str = ""):
        self.tg_token, self.tg_chat, self.discord = telegram_token, telegram_chat_id, discord_webhook

    def send(self, text: str) -> None:
        log.info("ALERT: %s", text)
        if self.tg_token and self.tg_chat:
            self._post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                       urllib.parse.urlencode({"chat_id": self.tg_chat, "text": text[:4000]}).encode(),
                       "application/x-www-form-urlencoded")
        if self.discord:
            self._post(self.discord, json.dumps({"content": text[:1900]}).encode(), "application/json")

    @staticmethod
    def _post(url: str, body: bytes, ctype: str) -> None:
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype, "User-Agent": "trading-agent"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # alerts must never crash the trading loop
            log.warning("notification failed: %s", e)
