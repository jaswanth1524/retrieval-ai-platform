"""Shared answers (api/shares.py): share a conversation or an answer, open it by link.

Every route needs a key (any level): sharing an answer is like asking, and so is
reading one. A share is opened by its id alone, from any workspace, but only with a key
that may use the workspace it was made in — otherwise it is a 404, as if it didn't
exist. Deleting one takes the key that made it, or a full key.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response

from api.audit import audit
from api.dependencies import get_share_store, request_grant, require_api_key
from api.keys import KeyGrant
from api.routes.deps import SettingsDep
from api.schemas import (
    ShareCreateRequest,
    SharedConversationResponse,
    SharedTurn,
    ShareListResponse,
    ShareSummaryResponse,
)
from api.shares import Share, ShareLimitError, ShareStore
from api.workspaces import workspace_of

router = APIRouter()

ShareStoreDep = Annotated[ShareStore | None, Depends(get_share_store)]
GrantDep = Annotated[KeyGrant, Depends(request_grant)]


def _store(store: ShareStore | None) -> ShareStore:
    if store is None:
        raise HTTPException(status_code=404, detail="Sharing is off (SHARES_ENABLED).")
    return store


def _summary(share: Share) -> ShareSummaryResponse:
    return ShareSummaryResponse(
        id=share.id,
        title=share.title,
        workspace=share.workspace,
        created_at=share.created_at,
        expires_at=share.expires_at,
        created_by=share.created_by,
        turns=share.turns,
    )


def _readable(store: ShareStore, share_id: str, grant: KeyGrant) -> Share:
    share = store.get(share_id) if len(share_id) <= 64 else None
    # One answer for "no such share", "expired" and "not for this key": the id is the
    # only thing a stranger holds, and it should tell them nothing.
    if share is None or not grant.can_use("" if share.workspace == "default" else share.workspace):
        raise HTTPException(status_code=404, detail="No such share (or it has expired).")
    return share


@router.post(
    "/shares",
    response_model=ShareSummaryResponse,
    status_code=201,
    dependencies=[Depends(require_api_key)],
    responses={409: {"description": "The workspace holds SHARE_MAX_PER_WORKSPACE shares."}},
)
def create_share(
    request: ShareCreateRequest, settings: SettingsDep, store: ShareStoreDep, grant: GrantDep
) -> ShareSummaryResponse:
    shares = _store(store)
    try:
        share = shares.add(
            workspace=workspace_of(settings),
            created_by=grant.name,
            title=request.title,
            turns=[turn.model_dump() for turn in request.turns],
        )
    except ShareLimitError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"This workspace already holds {settings.share_max_per_workspace} shares: "
            "delete some first.",
        ) from exc
    audit("share.create", workspace=workspace_of(settings), share=share.id, turns=share.turns)
    return _summary(share)


@router.get(
    "/shares",
    response_model=ShareListResponse,
    dependencies=[Depends(require_api_key)],
)
def list_shares(
    settings: SettingsDep, store: ShareStoreDep, grant: GrantDep, mine: bool = False
) -> ShareListResponse:
    """This workspace's shares, newest first (``mine``: only this key's)."""

    shares = _store(store).list(workspace_of(settings), created_by=grant.name if mine else None)
    return ShareListResponse(shares=[_summary(share) for share in shares])


@router.get(
    "/shares/{share_id}",
    response_model=SharedConversationResponse,
    dependencies=[Depends(require_api_key)],
)
def read_share(share_id: str, store: ShareStoreDep, grant: GrantDep) -> SharedConversationResponse:
    share = _readable(_store(store), share_id, grant)
    turns = [SharedTurn.model_validate(turn) for turn in share.payload or []]
    return SharedConversationResponse(**_summary(share).model_dump(), conversation=turns)


@router.delete(
    "/shares/{share_id}",
    status_code=204,
    dependencies=[Depends(require_api_key)],
)
def delete_share(share_id: str, store: ShareStoreDep, grant: GrantDep) -> Response:
    shares = _store(store)
    share = _readable(shares, share_id, grant)
    if grant.level != "full" and grant.name != share.created_by:
        raise HTTPException(
            status_code=403, detail="Only the key that shared it, or a full key, can delete it."
        )
    shares.delete(share.id)
    audit("share.delete", workspace=share.workspace, share=share.id)
    return Response(status_code=204)


def register(app: FastAPI) -> None:
    app.include_router(router)
