"""Normalize raw clone comments: TipTap richtext -> plain text, attached bets.

Comment JSON (verified against the live clone): `content` is a TipTap/ProseMirror
document; trade-justification comments carry the bet inline (betId / betOutcome /
betAmount); threaded replies carry replyToCommentId.
"""

from __future__ import annotations

from typing import Any

Comment = dict[str, Any]

# Block-level TipTap node types that should end with a line break in plain text.
_BLOCK_NODES = {"paragraph", "heading", "blockquote", "listItem", "codeBlock"}


def comment_text(comment: Comment) -> str:
    """Extract plain text from a comment's TipTap `content` document."""
    content = comment.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        if node.get("text"):
            parts.append(str(node["text"]))
        walk(node.get("content") or [])
        if node.get("type") in _BLOCK_NODES:
            parts.append("\n")

    walk(content)
    return "".join(parts).strip()


def attached_bet(comment: Comment) -> dict | None:
    """The bet a trade-justification comment carries inline, or None."""
    if not comment.get("betId"):
        return None
    return {
        "bet_id": comment["betId"],
        "outcome": comment.get("betOutcome"),
        "amount": comment.get("betAmount"),
        "limit_prob": comment.get("betLimitProb"),
    }


def is_reply(comment: Comment) -> bool:
    return bool(comment.get("replyToCommentId"))


def author(comment: Comment) -> str:
    return comment.get("userUsername") or comment.get("userName") or "?"
