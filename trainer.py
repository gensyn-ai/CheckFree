import random
import time
from deccom.protocols.peerdiscovery.kademliadiscovery import KademliaDiscovery
from sys import argv
import asyncio
from deccom.cryptofuncs.hash import SHA256
from communication.pp_protocol import PPProtocl
from deccom.nodes import StreamNode, Node
from deccom.protocols.defaultprotocol import DefaultProtocol
from deccom.peers import Peer
from deccom.protocols.streamprotocol import StreamProtocol
from multiprocessing import Lock, Process, Queue, current_process
import json
from communication.llm_subp import run_p
from deccom.protocols.delayprotocol import DelayProtocol
from pprint import pprint
import json
from communication.communication_costs import *
seq_l = 1024
n_layers = 4
batch_size = 8
dmodel = 1024
num_heads = 16



if __name__ == '__main__':
    curr_id = int(argv[1])
    setting = argv[2]
    f_rate = int(argv[3])
    
    with open(f"failure_p_configs/{f_rate}.json","r") as fd:
        failures = json.load(fd)

    prev = 0
    failures = iter(failures[str(curr_id)])
    loop = asyncio.new_event_loop()
    locations = get_locations()
    send_mbs = 6
    def delay_map(currid,otherid):
        p1 = locations[int(currid) % len(locations)]
        p2 = locations[int(otherid) % len(locations)]
        if DELAY_BANDWIDTHS.get(p1+"-"+p2) != None:
            ret = DELAY_BANDWIDTHS.get(p1+"-"+p2)
        elif DELAY_BANDWIDTHS.get(p2+"-"+p1) != None:
            ret = DELAY_BANDWIDTHS.get(p2+"-"+p1)
        else:
            ret = (10,0.500)
        # return (10,1.000)
        return (ret[0] - 0.1,ret[1])
    loc = locations[int(curr_id) % len(locations)]
    world = 21
    partitions = [
        [0,1,2],
        [3,4,5],
        [6,7,8],
        [9,10,11],
        [12,13,14],
        [15,16,17],
        [18,19,20]
    ]
    meshid = curr_id
    own_stage = curr_id // 3
    
    has_weights = True
    
    while True:
        my_peer  = Peer(None, pub_key=str(curr_id))
        port = None
        fail_at = next(failures) - prev
        prev += fail_at
        protocol = DefaultProtocol()
        gossip = KademliaDiscovery([],interval=30, always_split = True)
        gossip.set_lower(protocol)
        stream = StreamProtocol(False)
        stream.set_lower(gossip)
        delayer = DelayProtocol(delay_map,False)
        delayer.set_lower(stream)
        n = Peer(("127.0.0.1", 10015))
        if curr_id != 0:
            gossip.bootstrap_peers.append(n)
            time.sleep(1)

        queue_in = Queue(1024)
        queue_out = Queue(1024)
        device = "cuda"
        if meshid > 5:
            device = "cuda:1"
        if meshid > 10:
            device = "cuda:2"
        if meshid > 15:
            device = "cuda:3"
        # if curr_id > 9:
        #     device = "cuda:4"
        # if curr_id > 12:
        #     device = "cuda:5"
        # if curr_id > 15:
        #     device = "cuda:6"
        # if curr_id > 18:
        #     device = "cuda:7"
        cb = loop.create_future()
        subprocess = Process(target=run_p,args=(queue_out,queue_in,curr_id,own_stage,seq_l,n_layers,batch_size,dmodel,num_heads,send_mbs, device)) 
        trainingp = PPProtocl(stage=own_stage, meshid=meshid, 
                    MAX_STAGE=len(partitions), MAX_SEND = send_mbs, 
                    stage_size = len(partitions[0]), has_weights=has_weights,
                    queue_in=queue_in, queue_out=queue_out, subprocess=subprocess,
                    crash_callback=cb, fail_at=fail_at, strategy=setting)
        trainingp.set_lower(delayer)
        
        subprocess.start()
        
        me = StreamNode(my_peer, trainingp,ip_addr="127.0.0.1", port = 10015 if curr_id == 0 else port)
        
        print("run...")
        
        loop.run_until_complete(me.listen())
        loop.run_until_complete(cb)
        loop.run_until_complete(me.close())
        
        curr_id += 21
        
