import torch

from fairseq.data import encoders
from fairseq import bleu
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.modules.scst.generator import SimpleSequenceGenerator


@register_criterion('self_critical_sequence_training')
class SelfCriticalSequenceTrainingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task = task)
        self.args = args

        self.generator = SimpleSequenceGenerator(beam=args.scst_beam,
                                                 penalty=args.scst_penalty,
                                                 max_pos=args.max_target_positions,
                                                 eos_index=task.target_dictionary.eos_index)

        # Needed for decoding model output to string
        self.conf_tokenizer = encoders.build_tokenizer(args)
        self.conf_decoder = encoders.build_bpe(args)
        self.target_dict = task.target_dictionary

        # Tokenizer needed for computing CIDEr scores
        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)
 
        self.scorer = bleu.SacrebleuScorer()

        self.pad_idx = task.target_dictionary.pad()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--scst-beam', type=int, default=5,
                            help='beam size')
        parser.add_argument('--scst-penalty', type=float, default=1.0,
                            help='beam search length penalty')
        parser.add_argument('--scst-validation-set-size', type=int, default=0, metavar='N',
                            help='limited size of validation set')
        # parser.add_argument('--bpe', type=str, default=None,
        #                     help='bpe method for translated tokens')

    @property
    def get_ids(self):
        return self.task.dataset('train').img_ds.image_ids

    ####TO FIX... decoder(BPE); tokenizer; 是否还需要detokenize??
    def decode(self, x):
        """Decode model output.
        """
        x = self.target_dict.string(x)
        x = self.conf_decoder.decode(x)
        if self.conf_tokenizer is not None:
            x = self.conf_tokenizer.decode(x)
        return x
    
    # @property
    # def get_ids(self):
    #     print(type(self.task.dataset('train')))
    #     print(self.task.dataset('train').keys())
    #     print(type(self.task.dataset('train').id))
    #     assert 0 
    #     return self.task.dataset('train').img_ds.image_ids

    def generate(self, model, sample):
        """Generate captions using (simple) beam search.
        """
        tgt_translations = dict()
        gen_translations = dict()

        scores, _, tokens, _ = self.generator.generate(model, sample)

        counter = 0
        for i, tb in enumerate(tokens):
            # sample_id = self.get_ids[i]
            sample_translations = self.decode(sample['target'][i])
            ##sample_translations: a sentence of string type
            for t in tb:
                counter += 1
                decoded = self.decode(t)
                tgt_translations[counter] = sample_translations
                gen_translations[counter] = decoded

        # if self.tokenizer.tokenize is not None:
        #     gen_translations = self.tokenizer.tokenize(gen_translations)
        return tgt_translations, gen_translations, scores

    def forward(self, model, sample, reduce=True):
        sample_indices = sample['id']
        sample_device = sample_indices.device

        tgt_translations, gen_translations, scores = self.generate(model, sample)

        reward = [[self.scorer.add_string(tgt_translations[i], gen_translations[i]),
                  self.scorer.score(), 
                  self.scorer.reset()][1] \
                  for i in range(1, (1+len(tgt_translations.keys())))]

        reward = torch.tensor(reward).to(device=sample_device).view(scores.shape)

        # Mean of rewards is used as baseline rather than greedy
        # decoding (see also https://arxiv.org/abs/1912.08226).
        reward_baseline = torch.mean(reward, dim=1, keepdim=True)

        loss = -scores * (reward - reward_baseline)
        loss = loss.mean()

        #### TO FIX...
        sample_nsentences = len(sample)

        y = [torch.sum(sample['target'][i,]!=self.pad_idx) for i in range(len(sample['target']))]        
        for i, _ in enumerate(y):
            if i==0:
                sample_ntokens = _
            else:
                sample_ntokens+=_

        logging_output = {
            'loss': loss.data,
            'ntokens': sample_ntokens,
            'nsentences': sample_nsentences,
            'sample_size': sample_nsentences,
        }
        return loss, sample_nsentences, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs),
            'ntokens': sum(log.get('ntokens', 0) for log in logging_outputs),
            'nsentences': sum(log.get('nsentences', 0) for log in logging_outputs),
            'sample_size': sum(log.get('sample_size', 0) for log in logging_outputs)
        }

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)
