class Logger():
  def __init__(self, print_level: int = 3):
    '''
    print_level: prints all messages with level <= print_level
    '''

    self.print_level = print_level
    self.messages = []
  
  def log(self, msg: str, level: int):
    self.messages.append(msg)

    do_print = level <= self.print_level
    
    if do_print:
      print(msg)
