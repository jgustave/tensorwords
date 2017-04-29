""" use a simple work queue store results to S3 """
import boto3
import argparse

def main():
    parser = argparse.ArgumentParser(description='Do Stuff')
    parser.add_argument('--foo', type=int,default = None)


    print("")
    args = parser.parse_args()
    print(args.foo )

    #s3 = boto3.resource('s3')
    #s3.Object('jd-wine-data', 'winemodel/hello.txt').put(Body=open('/tmp/hello.txt', 'rb'))
    pass


if __name__ == '__main__':
    main()

