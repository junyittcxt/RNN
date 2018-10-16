import numpy as np
import time
def main():
  x =2
  def m2():
    x = np.random.normal(0, 0.5,1000000000)
    time.sleep(5)

    print(x)
    print("done")
  m2()
  print(x)
  time.sleep(15)


if __name__ == "__main__":
  main()
