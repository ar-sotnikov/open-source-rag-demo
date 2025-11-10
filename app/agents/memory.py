from aiocache import Cache

# Instantiate the in-memory cache.
cache = Cache(Cache.MEMORY)
USER_ID = "test_user"


async def get_history(user_id: str) -> list[dict[str, str]]:
    """Retrieves the conversation memory for a given user_id."""
    history = await cache.get(user_id)
    return history if history is not None else []


async def add_message(user_id: str, role: str, content: str):
    """
    Adds a new message to a user's conversation memory.
    This performs the full "Read-Modify-Write" cycle.
    """
    # READ the current history
    history = await get_history(user_id)

    # MODIFY by appending the new message
    history.append({"role": role, "content": content})

    # WRITE the entire updated history back
    await cache.set(user_id, history)
