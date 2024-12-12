from pymilvus import connections

def drop_all_connections():
    """
    Drops all connections to the Milvus database.
    """
    for alias, conn in connections.list_connections():
        connections.remove_connection(alias)