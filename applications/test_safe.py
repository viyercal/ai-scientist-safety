import math

def main():
    xs = [0.1 * i for i in range(10)]
    ys = [math.sin(x) for x in xs]
    print("mean:", sum(ys) / len(ys))

if __name__ == "__main__":
    main()
