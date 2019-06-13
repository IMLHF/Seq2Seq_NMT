import sys


def printinfo(msg, f=None, new_line=True):
  if new_line:
    msg += '\n'
  print(msg, end='')
  if f:
    f = open(f,'a+')
    f.writelines(msg)
    f.close()
  sys.stdout.flush()

def print_hparams(short=True):
  import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.base_config.__dict__
    for key in sorted(supper_dict.keys()):
      if key in self_dict_keys:
        print('%s:%s' % (key,self_dict[key]))
      else:
        print('%s:%s' % (key,supper_dict[key]))
    print('--------------------------\n')
  print('Short hparams:')
  [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  print('--------------------------\n')


if __name__ == '__main__':
  print_hparams(False)
  # printinfo('testsmse')
  # printinfo('testmsfd','test',False)
  # printinfo('testmsfd','test')
