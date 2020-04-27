import os

class Path():
  
  def __init__(self, path: str):
    self.path = path
  
  def ensure_created(self):
    os.makedirs(self.path, exist_ok=True)

  def get_path(self):
    return self.path

  def join(self, filename):
    result = os.path.join(self.path, filename)
    return result
