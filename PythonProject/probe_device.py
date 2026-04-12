# Overwrite to log to probe_out.txt for reliable capture
import depthai as dai
import serial.tools.list_ports as lp
import traceback
import sys, os

out_path = 'probe_out.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    def w(*a, **kw):
        print(*a, file=f, **kw)
    try:
        w('PY:', sys.executable)
        w('CWD:', os.getcwd())

        try:
            w('depthai __file__:', getattr(dai, '__file__', None))
            w('depthai __version__:', getattr(dai, '__version__', None))
            w('dai attrs sample:', [a for a in dir(dai) if not a.startswith('_')][:200])
            w('has Device:', hasattr(dai, 'Device'))
            if hasattr(dai, 'Device'):
                w('Device members sample:', [a for a in dir(dai.Device) if not a.startswith('_')][:200])

            for fn in ('getAllAvailableDevices', 'getAllConnectedDevices', 'getAllOnboardDevices'):
                try:
                    if hasattr(dai.Device, fn):
                        try:
                            w(fn, '->', getattr(dai.Device, fn)())
                        except Exception as e:
                            w(fn, 'raised', repr(e))
                except Exception as e:
                    w('error checking', fn, repr(e))

            try:
                w('Attempting dai.Device() ...')
                d = dai.Device()
                try:
                    plat = getattr(d.getPlatform(), 'name', None)
                    w('Device created ok, platform =', plat)
                    try:
                        d.close()
                        w('Device closed successfully')
                    except Exception as e:
                        w('Device close failed:', repr(e))
                except Exception as e:
                    w('Device instance op failed:', repr(e))
            except Exception as e:
                w('dai.Device() error:', repr(e))
                traceback.print_exc(file=f)

        except Exception as e:
            w('Import or initial checks failed:', repr(e))
            traceback.print_exc(file=f)

    except Exception as e:
        # If writing itself fails, fall back to stdout
        print('Probe write failed:', e)

    # Serial ports
    try:
        ports = list(lp.comports())
        w('Serial ports count:', len(ports))
        for p in ports:
            w(' -', p.device, p.description)
    except Exception as e:
        w('serial listing error:', repr(e))

w('\n--- probe done ---')
