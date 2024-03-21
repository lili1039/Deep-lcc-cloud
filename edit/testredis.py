import redis
import pickle

rs = redis.StrictRedis(host='127.0.0.1',db=2, port=6379,password="chlpw1039")


rs.mset({'test0':0})
rs.mset({'test1':pickle.dumps([1,2,3])})


value = rs.mget('aaa')[0]

print(value)
rs.flushdb()