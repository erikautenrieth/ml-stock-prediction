from backend.infra.storage.dagshub_storage import DagsHubStorage


def get_remote_storage() -> DagsHubStorage:
    return DagsHubStorage()
