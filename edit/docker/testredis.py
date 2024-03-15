import redis
import pickle

rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")

rs.mset({f'wait_expect_state':pickle.dumps(0)})
rs.mset({f'wait_expect_state':pickle.dumps([1,2,3])})
value_bytes = rs.mget('wait_expect_state')[0]
value = pickle.loads(value_bytes)

print(value)