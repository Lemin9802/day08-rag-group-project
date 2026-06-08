def add_turn(history, role, content):
    history.append({"role": role, "content": content})
    return history


def last_turns(history, n=6):
    return history[-n:]


def clear_memory():
    return []
