import sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pigean.multi_y_covariance import load_batch_correlations

class ManifestTests(unittest.TestCase):
 def test_alignment_provenance_and_rejections(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);manifest=root/'manifest.tsv';genes=root/'genes';matrix=root/'r.npz'
   genes.write_text('B\nA\nC\n');R=np.array([[1,.2,.1],[.2,1,.3],[.1,.3,1.]])
   sparse.save_npz(matrix,sparse.csr_matrix(R))
   header='trait\tmatrix\tgenes\tresponse_scale\n';row='T\tr.npz\tgenes\tbinary\n';manifest.write_text(header+row)
   def load(**kwargs):return load_batch_correlations(manifest,['T'],['A','B','C'],response_scale=kwargs.get('scale','binary'),memory_bytes=kwargs.get('memory',10**7))
   matrices,metadata=load();np.testing.assert_allclose(matrices[0].toarray(),R[[1,0,2]][:,[1,0,2]])
   self.assertEqual(len(metadata[0]['matrix_sha256']),64)
   for kwargs in [{'scale':'linear'},{'memory':1}]:
    with self.assertRaises(ValueError):load(**kwargs)
   for bad in [header,header+row+row]:
    manifest.write_text(bad)
    with self.assertRaises(ValueError):load()
   manifest.write_text(header+row)
   for bad in ['A\nA\nC\n','B\nC\nD\n']:
    genes.write_text(bad)
    with self.assertRaises(ValueError):load()
   genes.write_text('B\nA\nC\n')
   # Structurally valid pairwise correlations that are not a valid joint correlation.
   sparse.save_npz(matrix,sparse.csr_matrix([[1,.9,.9],[.9,1,-.9],[.9,-.9,1]]))
   with self.assertRaisesRegex(ValueError,'positive semidefinite'):load()

if __name__=='__main__':unittest.main()
