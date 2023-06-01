def get_protein_sequence(uniprot_id):
    import pandas as pd
    import requests
    get_protein_sequence= pd.read_csv('/lustre/scratch123/hgi/projects/huvec/analysis/ml/qstar/data/uniprot-compressed_true_download_true_fields_accession_2Cprotein_nam-2023.05.29-17.52.20.88.tsv.gz',compression='gzip',sep='\t')
    try:
        sequence = get_protein_sequence[get_protein_sequence['Entry']==uniprot_id]['Sequence'].values[0]
    except:
        # this must be an isoform not part of the retrieved uniprot ids.
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.ok:
            # Extract the protein sequence from the response content
            lines = response.text.split("\n")
            sequence = "".join(lines[1:])  # Join all lines except the first (header)
            return sequence
        else:
            print(f"Error retrieving protein sequence for UniProt ID: {uniprot_id}")
            return None
    return sequence