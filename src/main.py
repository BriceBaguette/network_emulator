from network_emulator import NetworkEmulator
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('node_file', type=str,
                        help='Path to the topology node file')
    parser.add_argument('link_file', type=str,
                        help='Path to the topology node file')

    parser.add_argument('-r', '--generation_rate', type=int,
                        default=1000, help='Generation rate in packets per second (default: 1000)')
    parser.add_argument('-t', '--time', type=int,
                        default=10, help='Duration of the simulation in seconds')
    parser.add_argument('-n', '--num_generations', type=int,
                        default=1, help='Number of generations (default: 1)')
    parser.add_argument('-d', '--depth', type=int,
                        default=1, help='Depth of failure (default: 1)')
    parser.add_argument('-o', '--type', type=str,
                        default='resilience', help='type of test to run (default: resilience)')
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input file for the network')
    parser.add_argument('-l', '--load_folder', type=str, default=None, help='Folder to load the network')
    parser.add_argument('-s', '--save_folder', type=str, default=None, help='Folder to save the network')
    

    args = parser.parse_args()
    start = time.time()
    net_sim = NetworkEmulator(node_file=args.node_file,link_file=args.link_file ,generation_rate=args.generation_rate,
                              num_generation=args.num_generations, duration=args.time, max_fib_break=args.depth, input_file=args.input_file, load_folder=args.load_folder, save_folder=args.save_folder)
    net_sim.build()
    net_sim.start()
    if args.type == 'resilience':
        net_sim.network_resilience_testing()
    elif args.type == 'generate':
        net_sim.emulate_all()
    else:
        print("Invalid test type: either resilience or generate")
    #net_sim.emulate_all()
    end = time.time()
    print("Time taken: {}".format(end-start))

if __name__ == "__main__":
    main()
