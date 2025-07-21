import argparse
from datetime import datetime
import logging
import sys


VERSION = 1.0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

def process_args(args):
	parser = argparse.ArgumentParser(description='CLI scaffolding.')
	parser.add_argument('--version', action='version', version=str(VERSION))
	parser.add_argument('-f', '--flag', action='store_true',
		                help='flag desc')
	parser.add_argument('args', metavar='N', type=int, nargs=1,
	                    help='args desc')
	parser.add_argument('--option', dest='option1_dest',
		                help='option1 desc')

	return parser.parse_args(args)

def get_datestr():
	return datetime.now().strftime('%Y-%m-%d')

def main(args):
	logger.info("Command line args: {}".format(args))

	logger.info("Current datestr: {}".format(get_datestr()))

if __name__ == "__main__":
	args = process_args(sys.argv[1:])
	main(args)