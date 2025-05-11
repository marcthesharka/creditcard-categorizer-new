import os
import redis
from rq import Worker, Queue, Connection

listen = ['default']
redis_url = os.environ.get("STACKHERO_REDIS_URL")  # or REDISGREEN_URL, etc.
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
