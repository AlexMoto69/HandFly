import depthai as dai
import time
import concurrent.futures
import sys

PALM_MODEL_SLUG = "luxonis/mediapipe-palm-detection:192x192"
HAND_MODEL_SLUG = "luxonis/mediapipe-hand-landmarker:224x224"

def fetch_model(slug, platform, timeout=15):
    desc = dai.NNModelDescription(slug)
    desc.platform = platform
    def _get():
        return dai.getModelFromZoo(desc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_get)
        try:
            res = fut.result(timeout=timeout)
            return True, res
        except concurrent.futures.TimeoutError:
            return False, 'timeout'
        except Exception as e:
            return False, repr(e)

print('PY:', sys.executable)
print('Starting device probe...')
try:
    d = dai.Device()
    platform = d.getPlatform().name
    print('Device platform:', platform)
    d.close()
except Exception as e:
    print('Device instantiation failed:', e)
    raise

print('Attempting to fetch palm model with 15s timeout...')
ok, r = fetch_model(PALM_MODEL_SLUG, platform, timeout=15)
print('Palm fetch:', ok, r)

print('Attempting to fetch hand model with 15s timeout...')
ok, r = fetch_model(HAND_MODEL_SLUG, platform, timeout=15)
print('Hand fetch:', ok, r)

print('Done.')

