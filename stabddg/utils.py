from Bio import PDB

def extract_chains(input_pdb, output1_pdb, chains1, output2_pdb, chains2):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb)

    io = PDB.PDBIO()

    # Save chains for first output
    class SelectChains1(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chains1

    io.set_structure(structure)
    io.save(output1_pdb, select=SelectChains1())

    # Save chains for second output
    class SelectChains2(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chains2

    io.set_structure(structure)
    io.save(output2_pdb, select=SelectChains2())