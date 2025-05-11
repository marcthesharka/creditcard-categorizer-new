import os
from redis import Redis
from rq import Worker, Queue

listen = ['default']
redis_url = (
    os.environ.get("STACKHERO_REDIS_URL_TLS") or
    os.environ.get("STACKHERO_REDIS_URL_CLEAR") or
    os.environ.get("REDISGREEN_URL") or
    os.environ.get("REDISCLOUD_URL") or
    os.environ.get("MEMETRIA_REDIS_URL")
)
if not redis_url:
    raise RuntimeError("No Redis URL found in environment variables! Please check your Heroku config vars.")
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    worker = Worker([Queue(name, connection=conn) for name in listen], connection=conn)
    worker.work()
