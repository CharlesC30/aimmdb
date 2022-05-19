from fastapi import HTTPException

from tiled.adapters.mapping import MapAdapter
from tiled.utils import SpecialUsers, import_object

READ = object()  # sentinel
WRITE = object()  # sentinel


def require_write_permission(method):
    def inner(self, *args, **kwargs):
        if WRITE not in self.permissions:
            raise HTTPException(
                status_code=403, detail="principal does not have write permission"
            )
        else:
            return method(self, *args, **kwargs)

    return inner


class AIMMAccessPolicy:
    def __init__(self, access_lists, *, provider):
        self.access_lists = {}
        self.provider = provider
        for key, value in access_lists.items():
            value_set = set(value)
            if not value_set.issubset({"r", "w"}):
                raise KeyError(
                    f"AIMMAccessPolicy: value {value} must be one of (r, w, rw)"
                )

            access = []
            if "r" in value_set:
                access.append(READ)
            if "w" in value_set:
                access.append(WRITE)

            self.access_lists[key] = access

    def get_id(self, principal):
        # Get the id (i.e. username) of this Principal for the
        # associated authentication provider.

        if principal is None:
            return None

        if isinstance(principal, SpecialUsers):
            return principal

        for identity in principal.identities:
            if identity.provider == self.provider:
                return identity.id
        else:
            raise ValueError(
                f"Principcal {principal} has no identity from provider {self.provider}. "
                f"Its identities are: {principal.identities}"
            )

    def permissions(self, principal):
        id = self.get_id(principal)
        if id is SpecialUsers.admin:
            return {READ, WRITE}
        else:
            return self.access_lists.get(id, None)
