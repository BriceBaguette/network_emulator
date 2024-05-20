from network_emulator import NetworkEmulator
import argparse


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
    

    args = parser.parse_args()

    print("Node file:", args.node_file)
    print("Link file:", args.link_file)
    print("Generation rate:", args.generation_rate)
    print("Number of generations:", args.num_generations)
    net_sim = NetworkEmulator(node_file=args.node_file,link_file=args.link_file ,generation_rate=args.generation_rate,
                              num_generation=args.num_generations, duration=args.time, max_fib_break=args.depth)
    net_sim.build()
    net_sim.start()
    if args.type == 'resilience':
        net_sim.network_resilience_testing()
    elif args.type == 'generate':
        net_sim.emulate("kAAAAAAA","kAAAAAAE")
    else:
        print("Invalid test type: either resilience or generate")
    #net_sim.emulate_all()


if __name__ == "__main__":
    main()