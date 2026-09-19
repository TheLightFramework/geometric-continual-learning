import tempfile
from pathlib import Path
import unittest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from gcl.model import Config,Model
from gcl.io import save,load,setup,tensor_hash,object_hash

class ModelTests(unittest.TestCase):
    def setUp(self):
        setup(11)
        self.m=Model(Config(width=16,depth=2,heads=2,cell_rank=8,triangle_rank=4,tetra_rank=4,vocab=32,valid_vocab=32,max_length=32),11).eval()
        self.x=torch.tensor([[1,2,3,4]])

    def test_cached_logits(self):
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            full=self.m(self.x)['logits'][:,-1]
            a=self.m(self.x[:,:2],use_cache=True)
            b=self.m(self.x[:,2:],cache=a['cache'],use_cache=True)['logits'][:,-1]
        torch.testing.assert_close(full,b,atol=1e-6,rtol=1e-5)

    def test_stale_cache(self):
        a=self.m(self.x,use_cache=True);self.m.stepped()
        with self.assertRaisesRegex(ValueError,'Stale'): self.m(self.x,cache=a['cache'])

    def test_save_reload(self):
        with tempfile.TemporaryDirectory() as d:
            save(self.m,Path(d)/'model');m,_=load(Path(d)/'model')
            self.assertEqual(tensor_hash(m.state_dict()),tensor_hash(self.m.state_dict()))
            self.assertEqual(object_hash(m.topology()),object_hash(self.m.topology()))
            torch.testing.assert_close(m(self.x)['logits'],self.m(self.x)['logits'],atol=0,rtol=0)

    def test_restricted_update(self):
        protected={n:p.detach().clone() for n,p in self.m.named_parameters() if not n.startswith('blocks.1.')}
        for n,p in self.m.named_parameters():p.requires_grad_(n.startswith('blocks.1.'))
        before=tensor_hash(self.m.state_dict());self.m.train()
        opt=torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad],lr=1e-4,weight_decay=0)
        loss=self.m(self.x,labels=torch.tensor([[2,3,4,5]]))['loss'];loss.backward();opt.step();self.m.stepped()
        self.assertNotEqual(before,tensor_hash(self.m.state_dict()))
        for n,p in self.m.named_parameters():
            if n in protected:self.assertTrue(torch.equal(protected[n],p))

    def test_zero_output_growth(self):
        before=self.m(self.x)['logits'].detach()
        self.m.birth(1,kind='LIFT',support=(0,1,2,3),rank=4,seed=3)
        torch.testing.assert_close(self.m(self.x)['logits'],before,atol=0,rtol=0)

    def test_lesion_restores(self):
        before=self.m(self.x)['logits'].detach();state=tensor_hash(self.m.state_dict());topology=object_hash(self.m.topology())
        with self.m.intervention(1,vertices=(0,)):
            self.assertFalse(torch.equal(self.m(self.x)['logits'],before))
        self.assertEqual(state,tensor_hash(self.m.state_dict()));self.assertEqual(topology,object_hash(self.m.topology()))
        torch.testing.assert_close(self.m(self.x)['logits'],before,atol=0,rtol=0)

    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError,'Context exceeded'):self.m(torch.ones((1,33),dtype=torch.long))

if __name__=='__main__':unittest.main()
