from network_emulator.network_emulator import NetworkEmulator
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Network Emulator')
    parser.add_argument('-nf', '--node_file',default=None ,type=str,
                        help='Path to the topology node file')
    parser.add_argument('-lf', '--link_file',default= None,type=str,
                        help='Path to the topology node file')
    parser.add_argument('-sf', '--single_file',default=None, type=str, help='Path to the topology file')
    parser.add_argument('-r', '--generation_rate', type=int,
                        default=1000, help='Generation rate in packets per second (default: 1000)')
    parser.add_argument('-n', '--num_generations', type=int,
                        default=1, help='Number of generations (default: 1)')
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input file for the network')
    parser.add_argument('-l', '--load_folder', type=str, default=None, help='Folder to load the network')
    parser.add_argument('-s', '--save_folder', type=str, default=None, help='Folder to save the network')
    

    args = parser.parse_args()
    start = time.time()
    net_sim = NetworkEmulator(node_file=args.node_file,link_file=args.link_file,single_file=args.single_file,generation_rate=args.generation_rate,
                              num_generation=args.num_generations,load_folder=args.load_folder, save_folder=args.save_folder)
    net_sim.build()
    net_sim.start()
    #net_sim.ecmp_analysis("1.1.1.1", show=True)
    #net_sim.all_latency_test()
    net_sim.ipm_session("1", "5")
    #net_sim.add_hw_issue(400,1000,"6569f30442e7f25d7a592660")
    #net_sim.all_ipm_session()
    #net_sim.hw_issue_detection(sink_measure_file="./src/results/sink.csv", source_measure_file="./src/results/source.csv",latency=True)

    end = time.time()
    print("Time taken: {}".format(end-start))

if __name__ == "__main__":
    main()
