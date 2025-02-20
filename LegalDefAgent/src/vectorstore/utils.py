from pymilvus import connections, Collection


def drop_all_connections():
    """
    Drops all connections to the Milvus database.
    """
    for alias, _ in connections.list_connections():
        connections.remove_connection(alias)


def connect_to_milvus(uri):
    """
    Connect to the Milvus database.

    Args:
        uri (str): The URI of the Mil
    """
    connections.connect(uri=uri)


def get_collection(collection_name):
    """
    Get the collection from the Milvus database.

    Returns:
        Collection: The collection from the Milvus database.
    """
    return Collection(collection_name)
