import sounddevice as sd
try:
   print(sd.query_devices())
except Exception as e:
   print("Ошибка при опросе устройств:", e)
