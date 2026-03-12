from netqmpi.sdk.communicator.communicator import QMPICommunicator

def print_info(message, rank):
    print(f"rank_{rank}: {message}")

def main(comm: QMPICommunicator = None):
    rank = comm.get_rank()
    size = comm.get_size()
    ROOT_RANK = 0

    with comm:
        # 1. Cada nodo aporta 1 solo cúbit
        local_qubits = [comm.create_qubit()]
        for q in local_qubits:
            q.H()

        # 2. Recolectamos los cúbits en el nodo ROOT
        full_qubits = comm.qgather(local_qubits, ROOT_RANK)

        # >>> ¡ESTE FLUSH ES VITAL! <<<
        # Obliga al rank_1 a enviar el cúbit antes de esperar
        comm.flush() 

        # 3. Procesamos los datos
        if rank == ROOT_RANK:
            values = []
            for q in full_qubits:
                values.append(q.measure())
            
            comm.flush()

            for i, val in enumerate(values):
                print_info(f"Qubit-{i} medido como: {val}", rank)
                
            # --- BARRERA DE SINCRONIZACIÓN (ROOT) ---
            for i in range(size):
                if i != ROOT_RANK:
                    comm.get_socket(ROOT_RANK, i).send_silent("done")
        else:
            # --- BARRERA DE SINCRONIZACIÓN (OTROS NODOS) ---
            comm.get_socket(rank, ROOT_RANK).recv_silent()

        # Flush final de seguridad para todos antes de salir
        comm.flush()

if __name__ == "__main__":
    main()
