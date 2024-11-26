from datetime import datetime
import pytz

def get_run_name():
  japan_tz = pytz.timezone('Asia/Tokyo')
  current_time_jst = datetime.now(japan_tz).strftime("%Y%m%d-%H%M%S")
  run_name = f"run-{current_time_jst}"

