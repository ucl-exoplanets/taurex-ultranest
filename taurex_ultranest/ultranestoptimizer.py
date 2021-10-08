
import ultranest
import ultranest.stepsampler
import time
from taurex.optimizer import Optimizer
import numpy as np
from taurex.util.util import quantile_corner, \
                                recursively_save_dict_contents_to_output
class UltranestSampler(Optimizer):

    def __init__(self, 
                 observed=None, model=None,
                 sigma_fraction=0.1,
                 num_live_points=500,
                 dlogz=0.5,
                 dkl=0.5,
                 frac_remain=0.01,
                 cluster_num_live_points=40,
                 max_num_improvement_loops=-1,
                 stepsampler='default',
                 nsteps=10,
                 step_scale=1.0,
                 adaptive_nsteps=False,
                 region_filter=False,
                 resume='subfolder', 
                 log_dir=None,
                 num_test_samples=2,
                 draw_multiple=True,
                 num_bootstraps=30,
                 ndraw_min=128,
                 ndraw_max=65536,
                 storage_backend='hdf5',
                 warmstart_max_tau=- 1):
        super().__init__('Ultranest', observed, model, sigma_fraction)

        self.num_live_points = int(num_live_points)
        self.dlogz = dlogz
        self.dkl = dkl
        self.frac_remain = frac_remain
        self.cluster_num_live_points = int(cluster_num_live_points)
        self.max_num_improvement_loops = int(max_num_improvement_loops)
        self.stepsampler = stepsampler
        self.nsteps = int(nsteps)
        self.step_scale = step_scale
        self.adaptive_nsteps = adaptive_nsteps
        self.region_filter = region_filter

        if self.stepsampler is None:
            self.stepsampler = 'default'

        self.resume = resume
        self.log_dir = log_dir
        self.num_test_samples = num_test_samples
        self.draw_multiple = draw_multiple
        self.num_bootstraps = int(num_bootstraps)
        self.ndraw_min = int(ndraw_min)
        self.ndraw_max = int(ndraw_max)
        self.storage_backend = storage_backend
        self.warmstart_max_tau = warmstart_max_tau





    def compute_fit(self):
        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)

        def ultranest_loglike(params):
            # log-likelihood function called by multinest
            fit_params_container = np.array(params)
            chi_t = self.chisq_trans(fit_params_container, data, datastd)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike

        def ultranest_prior(theta):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            cube = []

            for idx, prior in enumerate(self.fitting_priors):

                cube.append(prior.sample(theta[idx]))

            return np.array(cube)

        sampler = ultranest.ReactiveNestedSampler(self.fit_names,
                                                  ultranest_loglike,
                                                  transform=ultranest_prior,
                                                  resume=self.resume,
                                                  log_dir=self.log_dir,
                                                  num_test_samples=self.num_test_samples,
                                                  draw_multiple=self.draw_multiple,
                                                  num_bootstraps=self.num_bootstraps,
                                                  vectorized=False,
                                                  storage_backend=self.storage_backend,
                                                  warmstart_max_tau=self.warmstart_max_tau)

        sampler_choice = {
            'cube-mh': ultranest.stepsampler.CubeMHSampler,
            'region-mh': ultranest.stepsampler.RegionMHSampler,
            'cube-slice': ultranest.stepsampler.CubeSliceSampler,
            'region-slice': ultranest.stepsampler.RegionSliceSampler,
            'region-sequentical-slice': ultranest.stepsampler.RegionSequentialSliceSampler,
            'ball-slice': ultranest.stepsampler.BallSliceSampler,
            'region-ball-slice': ultranest.stepsampler.RegionBallSliceSampler,
        }

        stepchoice = self.stepsampler.lower()

        if stepchoice is not None and stepchoice in sampler_choice:
            sampler.stepsampler = sampler_choice[stepchoice](int(self.nsteps),
                                                             scale=self.step_scale,
                                                             adaptive_nsteps=self.adaptive_nsteps,
                                                             region_filter=self.region_filter)
        
        t0 = time.time()

        result = sampler.run(dlogz=self.dlogz, dKL=self.dkl, 
                             frac_remain=self.frac_remain, 
                             max_num_improvement_loops=self.max_num_improvement_loops,
                             min_num_live_points=self.num_live_points,
                             cluster_num_live_points=self.cluster_num_live_points,
                             )
        t1 = time.time()

        self.warning("Time taken to run 'Ultranest' is %s seconds", t1-t0)

        self.warning('Fit complete.....')

        self.ultranest_output = self.store_ultranest_output(result)

    def store_ultranest_output(self, result):
        """
        This turns the output fron ultranest into a dictionary that can
        be output by Taurex

        Parameters
        ----------
        result: :obj:`dict`
            Result from a ultranest sample call

        Returns
        -------
        dict
            Formatted dictionary for output

        """

        from tabulate import tabulate

        ultranest_output = {}
        ultranest_output['Stats'] = {}
        ultranest_output['Stats']['Log-Evidence'] = result['logz']
        ultranest_output['Stats']['Log-Evidence-Error'] = result['logzerr']
        ultranest_output['Stats']['H'] = result['H']
        ultranest_output['Stats']['Herror'] = result['Herr']

        fit_param = self.fit_names

        samples = result['weighted_samples']['points']
        weights = result['weighted_samples']['weights']
        logl = result['weighted_samples']['logl']
        mean, cov = result['posterior']['mean'], result['posterior']['stdev']
        ultranest_output['solution'] = {}
        ultranest_output['solution']['samples'] = samples
        ultranest_output['solution']['weights'] = weights
        ultranest_output['solution']['logl'] = logl
        # ultranest_output['solution']['covariance'] = cov
        ultranest_output['solution']['fitparams'] = {}

        max_weight = weights.argmax()

        table_data = []
        

        for idx, param_name in enumerate(fit_param):
            param = {}
            param['mean'] = mean[idx]
            param['sigma'] = cov[idx]
            trace = samples[:, idx]
            q_16, q_50, q_84 = quantile_corner(trace, [0.16, 0.5, 0.84],
                                               weights=np.asarray(weights))
            param['value'] = q_50
            param['sigma_m'] = q_50-q_16
            param['sigma_p'] = q_84-q_50
            param['trace'] = trace
            param['map'] = trace[max_weight]
            table_data.append((param_name, q_50, q_50-q_16))

            ultranest_output['solution']['fitparams'][param_name] = param

        return ultranest_output

    def get_samples(self, solution_idx):
        return self.ultranest_output['solution']['samples']

    def get_weights(self, solution_idx):
        return self.ultranest_output['solution']['weights']

    def write_fit(self, output):
        fit = super().write_fit(output)

        if self.ultranest_output:
            recursively_save_dict_contents_to_output(
                output, self.ultranest_output)

        return fit

    def chisq_trans(self, fit_params, data, datastd):
        res = super().chisq_trans(fit_params, data, datastd)

        if not np.isfinite(res):
            return 1e20
        
        return res


    def get_solution(self):
        """

        Generator for solutions and their
        median and MAP values

        Yields
        ------

        solution_no: int
            Solution number (always 0)

        map: :obj:`array`
            Map values

        median: :obj:`array`
            Median values

        extra: :obj:`list`
            Returns Statistics, fitting_params, raw_traces and
            raw_weights

        """

        names = self.fit_names
        opt_map = self.fit_values
        opt_values = self.fit_values
        for k, v in self.ultranest_output['solution']['fitparams'].items():
            # if k.endswith('_derived'):
            #     continue
            idx = names.index(k)
            opt_map[idx] = v['map']
            opt_values[idx] = v['value']

        yield 0, opt_map, opt_values, [('Statistics', self.ultranest_output['Stats']),
                                       ('fit_params',
                                        self.ultranest_output['solution']['fitparams']),
                                       ('tracedata',
                                        self.ultranest_output['solution']['samples']),
                                       ('weights', self.ultranest_output['solution']['weights'])]


    @classmethod
    def input_keywords(self):
        return ['ultranest', ]

    BIBTEX_ENTRIES = [
        """
        @article{Buchner_2014,
            title={A statistical test for Nested Sampling algorithms},
            volume={26},
            ISSN={1573-1375},
            url={http://dx.doi.org/10.1007/s11222-014-9512-y},
            DOI={10.1007/s11222-014-9512-y},
            number={1-2},
            journal={Statistics and Computing},
            publisher={Springer Science and Business Media LLC},
            author={Buchner, Johannes},
            year={2014},
            month={Sep},
            pages={383–392}
        }
       """,
        """
        @ARTICLE{2019PASP..131j8005B,
            author = {{Buchner}, Johannes},
                title = "{Collaborative Nested Sampling: Big Data versus Complex Physical Models}",
            journal = {\pasp},
                year = 2019,
                month = oct,
            volume = {131},
            number = {1004},
                pages = {108005},
                doi = {10.1088/1538-3873/aae7fc},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2019PASP..131j8005B},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """
    ]