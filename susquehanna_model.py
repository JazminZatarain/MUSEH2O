
import os
import numpy as np

import rbf_functions
import utils
# from rbf import RBF
from numba import njit


def create_path(rest):
    # FIXME my dir is now retreived repeatedly
    my_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(my_dir, rest))


class SusquehannaModel:
    gammaH20 = 1000.0
    GG = 9.81
    n_days_in_year = 365

    def __init__(self, l0, l0_muddy_run, d0, n_years, rbf, rbf_kwargs,
                 historic_data=True):
        """

        Parameters
        ----------
        l0 : float
             initial condition Conowingo
        l0_muddy_run : float
                       initial condition Muddy Run
        d0 : int
             intial start date
        n_years : int
        rbf : callable
        rbf_kwargs : dict
        historic_data : bool, optional
                        if true use historic data, if false use stochastic
                        data

        """

        self.init_level = l0  # feet
        self.init_level_MR = l0_muddy_run
        self.day0 = d0

        # FIXME: logging should not be handled through a boolean
        # FIXME: seems log is mislabeled, just a trace during runtime
        # but by simply setting a logger in the main run
        self.log_level_release = False

        # variables from the header file
        self.input_min = []
        self.input_max = []
        self.output_max = []
        self.rbf = rbf
        self.rbf_kwargs = rbf_kwargs

        # log level / release
        self.blevel_CO = []
        self.blevel_MR = []
        self.ratom = []
        self.rbalt = []
        self.rches = []
        self.renv = []

        # historical record #1000 simulation horizon (1996,2001)
        self.n_years = n_years
        self.time_horizon_H = self.n_days_in_year * self.n_years
        self.hours_between_decisions = 4  # 4-hours decision time step
        self.decisions_per_day = int(24 / self.hours_between_decisions)
        self.n_days_one_year = 365

        # Constraints for the reservoir
        self.min_level_chester = 99.8  # ft of water
        self.min_level_app = 98.5  # ft of water
        self.min_level_baltimore = 90.8  # ft of water
        self.min_level_conowingo = 100.5  # ft

        # n_days_one_year = 1*365 moved to init
        # Conowingo characteristics
        self.lsv_rel = utils.loadMatrix(
            create_path("./data1999/lsv_rel_Conowingo.txt"), 3, 10
        )  # level (ft) - Surface (acre) - storage (acre-feet) relationships
        self.turbines = utils.loadMatrix(
            create_path("./data1999/turbines_Conowingo2.txt"), 3, 13
        )  # Max-min capacity (cfs) - efficiency of Conowingo plant turbines
        self.tailwater = utils.loadMatrix(
            create_path("./data1999/tailwater.txt"), 2, 18
        )  # tailwater head (ft) - release flow (cfs)
        self.spillways = utils.loadMatrix(
            create_path("./data1999/spillways_Conowingo.txt"), 3, 8
        )
        # substitute with newConowingo1
        # level (ft) - max release (cfs) - min release (cfs) for level > 108 ft

        # Muddy Run characteristics
        self.lsv_rel_Muddy = utils.loadMatrix(
            create_path("./data1999/lsv_rel_Muddy.txt"), 3, 38
        )  # level (ft) - Surface (acre) - storage (acre-feet) relationships
        self.turbines_Muddy = utils.loadVector(
            create_path("./data1999/turbines_Muddy.txt"), 4
        )
        # Turbine-Pumping capacity (cfs) - efficiency of Muddy Run plant (
        # equal for the 8 units)

        self.historic_data = historic_data
        if historic_data:
            self.load_historic_data()
            self.evaluate = self.evaluate_historic
        else:
            self.load_stochastic_data()
            self.evaluate = self.evaluate_mc

        # objectives parameters
        self.energy_prices = utils.loadArrangeMatrix(
            create_path("./data1999/Pavg99.txt"), 24, self.n_days_one_year
        )  # energy prices ($/MWh)
        self.min_flow = utils.loadVector(
            create_path("./data1999/min_flow_req.txt"), self.n_days_one_year
        )  # FERC minimum flow requirements for 1 year (cfs)
        self.h_ref_rec = utils.loadVector(
            create_path("./data1999/h_rec99.txt"), self.n_days_one_year
        )  # target level for weekends in touristic season (ft)
        self.w_baltimore = utils.loadVector(
            create_path("./data1999/wBaltimore.txt"), self.n_days_one_year
        )  # water demand of Baltimore (cfs)
        self.w_chester = utils.loadVector(
            create_path("./data1999/wChester.txt"), self.n_days_one_year
        )  # water demand of Chester (cfs)
        self.w_atomic = utils.loadVector(
            create_path("./data1999/wAtomic.txt"), self.n_days_one_year
        )  # water demand for cooling the atomic power plant (cfs)

        # standardization of the input-output of the RBF release curve
        self.input_max.append(self.n_days_in_year * self.decisions_per_day - 1)
        self.input_max.append(120)

        self.output_max.append(utils.computeMax(self.w_atomic))
        self.output_max.append(utils.computeMax(self.w_baltimore))
        self.output_max.append(utils.computeMax(self.w_chester))
        self.output_max.append(85412)
        # max release = tot turbine capacity + spillways @ max storage

    def load_historic_data(self):
        self.evap_CO_MC = utils.loadMultiVector(
            create_path("./data_historical/vectors/evapCO_history.txt"),
                   self.n_years,
            self.n_days_one_year
        )  # evaporation losses (inches per day)
        self.inflow_MC = utils.loadMultiVector(
            create_path("./data_historical/vectors/MariettaFlows_history.txt"),
            self.n_years, self.n_days_one_year
        )  # inflow, i.e. flows at Marietta (cfs)
        self.inflowLat_MC = utils.loadMultiVector(
            create_path("./data_historical/vectors/nLat_history.txt"),
            self.n_years,
            self.n_days_one_year
        )  # lateral inflows from Marietta to Conowingo (cfs)
        self.evap_Muddy_MC = utils.loadMultiVector(
            create_path("./data_historical/vectors/evapMR_history.txt"),
            self.n_years,
            self.n_days_one_year
        )  # evaporation losses (inches per day)
        self.inflow_Muddy_MC = utils.loadMultiVector(
            create_path("./data_historical/vectors/nMR_history.txt"),
            self.n_years,
            self.n_days_one_year
        )

    def load_stochastic_data(self):
        # stochastic hydrology
        self.evap_CO_MC = utils.loadMatrix(
            "./dataMC/evapCO_MC.txt", self.n_years, self.n_days_one_year
        )  # evaporation losses (inches per day)
        self.inflow_MC = utils.loadMatrix(
            "./dataMC/MariettaFlows_MC.txt", self.n_years, self.n_days_one_year
        )  # inflow, i.e. flows at Marietta (cfs)
        self.inflowLat_MC = utils.loadMatrix(
            "./dataMC/nLat_MC.txt", self.n_years, self.n_days_one_year
        )  # lateral inflows from Marietta to Conowingo (cfs)
        self.evap_Muddy_MC = utils.loadMatrix(
            "./dataMC/evapMR_MC.txt", self.n_years, self.n_days_one_year
        )  # evaporation losses (inches per day)
        self.inflow_Muddy_MC = utils.loadMatrix(
            "./dataMC/nMR_MC.txt", self.n_years, self.n_days_one_year
        )  # inflow to Muddy Run (cfs)

    def set_log(self, log_objectives):
        if log_objectives:
            self.log_objectives = True
        else:
            self.log_objectives = False

    def get_log(self):
        return self.blevel_CO, self.blevel_MR, self.ratom, self.rbalt, \
               self.rches, self.renv

    def apply_rbf_policy(self, rbf_input, center, radius, weights):

        # normalize inputs
        n_inputs = self.rbf_kwargs['n_inputs']
        formatted_input = rbf_input / self.input_max

        # apply rbf
        u = self.rbf(formatted_input, center, radius, weights,
                     self.rbf_kwargs['n_rbf'])

        # scale back
        uu = []
        for i in range(0, self.rbf_kwargs['n_outputs']):
            uu.append(u[i] * self.output_max[i])
        return uu

    def evaluate_historic(self, var, opt_met=1):
        return self.simulate(var, self.inflow_MC, self.inflowLat_MC,
                             self.inflow_Muddy_MC, self.evap_CO_MC,
                             self.evap_Muddy_MC, opt_met)

    def evaluate_mc(self, var, opt_met=1):
        obj, Jhyd, Jatom, Jbal, Jche, Jenv, Jrec = [], [], [], [], [], [], []
        # MC simulations
        n_samples = 2
        for i in range(0, n_samples):
            Jhydropower, Jatomicpowerplant, Jbaltimore, Jchester, \
            Jenvironment, Jrecreation = self.simulate(
                var,
                self.inflow_MC,
                self.inflowLat_MC,
                self.inflow_Muddy_MC,
                self.evap_CO_MC,
                self.evap_Muddy_MC,
                opt_met,
            )
            Jhyd.append(Jhydropower)
            Jatom.append(Jatomicpowerplant)
            Jbal.append(Jbaltimore)
            Jche.append(Jchester)
            Jenv.append(Jenvironment)
            Jrec.append(Jrecreation)

        # objectives aggregation (minimax)
        obj.insert(0, utils.computePercentile(Jhyd, 99))
        obj.insert(1, utils.computePercentile(Jatom, 99))
        obj.insert(2, utils.computePercentile(Jbal, 99))
        obj.insert(3, utils.computePercentile(Jche, 99))
        obj.insert(4, utils.computePercentile(Jenv, 99))
        obj.insert(5, utils.computePercentile(Jrec, 99))
        return obj

    def storage_to_level(self, s, lake):
        # s : storage
        # lake : which lake it is at
        # gets triggered decision step * time horizon
        s_ = utils.cubicFeetToAcreFeet(s)
        if lake == 0:
            h = utils.interpolate_linear(self.lsv_rel_Muddy[2],
                                         self.lsv_rel_Muddy[0], s_)
        else:
            h = utils.interpolate_linear(self.lsv_rel[2], self.lsv_rel[0], s_)
        return h

    def level_to_storage(self, h, lake):
        if lake == 0:
            s = utils.interpolate_linear(self.lsv_rel_Muddy[0],
                                         self.lsv_rel_Muddy[2], h)
        else:
            s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[2], h)
        return utils.acreFeetToCubicFeet(s)

    def level_to_surface(self, h, lake):
        if lake == 0:
            s = utils.interpolate_linear(self.lsv_rel_Muddy[0],
                                         self.lsv_rel_Muddy[1], h)
        else:
            s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[1], h)
        return utils.acreToSquaredFeet(s)

    def tailwater_level(self, q):
        return utils.interpolate_linear(self.tailwater[0], self.tailwater[1],
                                        q)

    def muddyrun_pumpturb(self, day, hour, level_Co, level_MR):
        # Determines the pumping and turbine release volumes in a day based
        # on the hour and day of week for muddy run
        QP = 24800  # cfs
        QT = 32000  # cfs

        # active storage = sMR - deadStorage
        qM = (self.level_to_storage(level_MR, 0) -
              self.level_to_storage(470.0, 0)) / 3600
        qp = 0.0
        qt = 0.0
        if day == 0:  # sunday
            if hour < 5 or hour >= 22:
                qp = QP
        elif 1 <= day <= 4:  # monday to thursday
            if hour <= 6 or hour >= 21:
                qp = QP
            if (7 <= hour <= 11) or (17 <= hour <= 20):
                qt = min(QT, qM)
        elif day == 5:  # friday
            if (7 <= hour <= 11) or (17 <= hour <= 20):
                qt = min(QT, qM)
        elif day == 6:  # saturday
            if hour <= 6 or hour >= 22:
                qp = QP
        # water pumping stops to Muddy Run beyond this point.
        # However, according to the conowingo authorities 800 cfs will be
        # released as emergency credits in order to keep the facilities from
        # running
        # Q: The level in Conowingo impacts the pumping in Muddy Run. How?
        if level_Co < 104.7:  # if True cavitation problems in pumping
            qp = 0.0

        if level_MR < 470.0:
            qt = 0.0
        return qp, qt  # pumping, Turbine release

    def actual_release(self, uu, level_Co, day_of_year):
        # Check if it doesn't exceed the spillway capacity
        Tcap = 85412  # total turbine capacity (cfs)
        # maxSpill = 1242857.0 # total spillway combined (cfs)

        # minimum discharge values at APP, Balitomore, Chester and downstream
        qm_A = 0.0
        qm_B = 0.0
        qm_C = 0.0
        qm_D = 0.0

        # maximum discharge values. The max discharge can be as much as the
        # demand in that area
        qM_A = self.w_atomic[day_of_year]
        qM_B = self.w_baltimore[day_of_year]
        qM_C = self.w_chester[day_of_year]
        qM_D = Tcap

        # reservoir release constraints
        if level_Co <= self.min_level_app:
            qM_A = 0.0
        else:
            qM_A = self.w_atomic[day_of_year]

        if level_Co <= self.min_level_baltimore:
            qM_B = 0.0
        else:
            qM_B = self.w_baltimore[day_of_year]
        if level_Co <= self.min_level_chester:
            qM_C = 0.0
        else:
            qM_C = self.w_chester[day_of_year]

        if level_Co > 110.2:  # spillways activated
            qM_D = (
                    utils.interpolate_linear(self.spillways[0],
                                             self.spillways[1],
                                             level_Co) + Tcap
            )  # Turbine capacity + spillways
            qm_D = (
                    utils.interpolate_linear(self.spillways[0],
                                             self.spillways[1],
                                             level_Co) + Tcap
            )  # change to spillways[2]

        # different from flood model
        if level_Co < 105.5:
            qM_D = 0.0
        elif level_Co < 103.5:
            qM_A = 0.0
        elif level_Co < 100.5:
            qM_C = 0.0
        elif level_Co < 91.5:
            qM_B = 0.0

        # actual release
        rr = []
        rr.append(min(qM_A, max(qm_A, uu[0])))
        rr.append(min(qM_B, max(qm_B, uu[1])))
        rr.append(min(qM_C, max(qm_C, uu[2])))
        rr.append(min(qM_D, max(qm_D, uu[3])))
        return rr

    @staticmethod
    @njit
    def g_hydRevCo(r, h, day_of_year, hour0, GG, gammaH20, tailwater, turbines,
                   energy_prices):
        def interpolate_tailwater_level(X, Y, x):
            dim = len(X) - 1
            if x <= X[0]:
                y = (x - X[0]) * (Y[1] - Y[0]) / (X[1] - X[0]) + Y[0]
                return y
            elif x >= X[dim]:
                y = Y[dim] + (Y[dim] - Y[dim - 1]) / (X[dim] - X[dim - 1]) * (
                        x - X[dim])
                return y
            else:
                y = np.interp(x, X, Y)
            return y

        cubicFeetToCubicMeters = 0.0283  # 1 cf = 0.0283 m3
        feetToMeters = 0.3048  # 1 ft = 0.3048 m
        Nturb = 13
        g_hyd = []
        g_rev = []
        pp = []
        c_hour = len(r) * hour0
        for i in range(0, len(r)):
            deltaH = h[i] - interpolate_tailwater_level(tailwater[0],
                                                        tailwater[1], r[i])
            q_split = r[i]
            for j in range(0, Nturb):
                if q_split < turbines[1][j]:
                    qturb = 0.0
                elif q_split > turbines[0][j]:
                    qturb = turbines[0][j]
                else:
                    qturb = q_split
                q_split = q_split - qturb
                p = (
                        0.79
                        * GG
                        * gammaH20
                        * (cubicFeetToCubicMeters * qturb)
                        * (feetToMeters * deltaH)
                        * 3600
                        / (3600 * 1000)
                )  # assuming lower efficiency as in Exelon docs
                pp.append(p)
            g_hyd.append(np.sum(np.asarray(pp)))
            g_rev.append(np.sum(np.asarray(pp)) / 1000 * energy_prices[c_hour][
                day_of_year])
            pp.clear()
            c_hour = c_hour + 1
        Gp = np.sum(np.asarray(g_hyd))
        Gr = np.sum(np.asarray(g_rev))
        return Gp, Gr

    @staticmethod
    @njit
    def g_hydRevMR(qp, qr, hCo, hMR, day_of_year, hour0, GG, gammaH20,
                   turbines_Muddy, energy_prices):
        n_turb = 8
        cubic_feet_to_cubic_meters = 0.0283  # 1 cf = 0.0283 m3
        feet_to_meters = 0.3048  # 1 ft = 0.3048 m
        g_hyd = []
        g_pump = []
        g_rev = []
        g_revP = []
        c_hour = len(qp) * hour0
        pT = 0.0
        pP = 0.0
        for i in range(0, len(qp)):
            # net head
            deltaH = hMR[i] - hCo[i]
            # 8 turbines
            qp_split = qp[i]
            qr_split = qr[i]
            for j in range(0, n_turb):
                if qp_split < 0.0:
                    qpump = 0.0
                elif qp_split > turbines_Muddy[2]:
                    qpump = turbines_Muddy[2]
                else:
                    qpump = qp_split

                p_ = (turbines_Muddy[3]
                      * GG
                      * gammaH20
                      * (cubic_feet_to_cubic_meters * qpump)
                      * (feet_to_meters * deltaH)
                      * 3600
                      / (3600 * 1000))  # KWh/h
                pP = pP + p_
                qp_split = qp_split - qpump

                if qr_split < 0.0:
                    qturb = 0.0
                elif qr_split > turbines_Muddy[0]:
                    qturb = turbines_Muddy[0]
                else:
                    qturb = qr_split

                p = (turbines_Muddy[1]
                     * GG
                     * gammaH20
                     * (cubic_feet_to_cubic_meters * qturb)
                     * (feet_to_meters * deltaH)
                     * 3600
                     / (3600 * 1000))  # kWh/h
                pT = pT + p
                qr_split = qr_split - qturb

            g_pump.append(pP)
            g_revP.append(pP / 1000 * energy_prices[c_hour][day_of_year])
            pP = 0.0
            g_hyd.append(pT)
            g_rev.append(pT / 1000 * energy_prices[c_hour][day_of_year])
            pT = 0.0
            c_hour = c_hour + 1
        return g_pump, g_hyd, g_revP, g_rev

    def res_transition_h(self, s0, uu, n_sim, n_lat, ev, s0_mr, n_sim_mr,
                         ev_mr, day_of_year, day_of_week, hour0):
        HH = self.hours_between_decisions  # 4 hour horizon
        sim_step = 3600  # s/hour
        leak = 800  # cfs

        # Storages and levels of Conowingo and Muddy Run
        shape = (HH+1, )

        storage_Co = np.empty(shape)
        level_Co = np.empty(shape)
        storage_mr = np.empty(shape)
        level_mr = np.empty(shape)
        # Actual releases (Atomic Power plant, Baltimore, Chester, Dowstream)

        shape = (HH,)
        release_A = np.empty(shape)
        release_B = np.empty(shape)
        release_C = np.empty(shape)
        release_D = np.empty(shape)
        q_pump = np.empty(shape)
        q_rel = np.empty(shape)

        # initial conditions
        storage_Co[0] = s0
        storage_mr[0] = s0_mr
        c_hour = HH * hour0

        for i in range(0, HH):
            # compute level
            level_Co[i] = self.storage_to_level(storage_Co[i], 1)
            level_mr[i] = self.storage_to_level(storage_mr[i], 0)
            # Muddy Run operation
            q_pump[i], q_rel[i] = self.muddyrun_pumpturb(day_of_week,
                                                         int(c_hour),
                                                         level_Co[i],
                                                         level_mr[i])

            # Compute actual release
            rr = self.actual_release(uu, level_Co[i], day_of_year)
            release_A[i] = rr[0]
            release_B[i] = rr[1]
            release_C[i] = rr[2]
            release_D[i] = rr[3]

            # Q: Why is this being added?
            # FIXME: actual release as numpy array then simple sum over slice
            # FIXME: into rr
            WS = release_A[i] + release_B[i] + release_C[i]

            # Compute surface level and evaporation losses
            surface_Co = self.level_to_surface(level_Co[i], 1)
            evaporation_losses_Co = utils.inchesToFeet(ev) * surface_Co / \
                                    86400  # cfs
            surface_MR = self.level_to_surface(level_mr[i], 0)
            evaporation_losses_MR = utils.inchesToFeet(
                ev_mr) * surface_MR / 86400  # cfs

            # System Transition
            storage_mr[i + 1] = storage_mr[i] + sim_step * (q_pump[i] - q_rel[
                i] + n_sim_mr - evaporation_losses_MR)
            storage_Co[i + 1] = storage_Co[i] + sim_step * (
                    n_sim + n_lat - release_D[i] - WS - evaporation_losses_Co -
                    q_pump[i] + q_rel[i] - leak
            )
            c_hour = c_hour + 1

            # storage_mr[i + 1] = storage_mr[i] + sim_step * (q_pump[i] - q_rel[
            #     i] + n_sim_mr - evaporation_losses_MR)
            # storage_Co[i + 1] = storage_Co[i] + sim_step * (
            #         n_sim + n_lat - release_D[i] - WS - evaporation_losses_Co -
            #         q_pump[i] + q_rel[i] - leak
            # )

        sto_co = storage_Co[HH]
        sto_mr = storage_mr[HH]
        rel_a = utils.computeMean(release_A)
        rel_b = utils.computeMean(release_B)
        rel_c = utils.computeMean(release_C)
        rel_d = utils.computeMean(release_D)

        level_Co = np.asarray(level_Co)
        # 4-hours hydropower production/revenue
        # hp = self.g_hydRevCo(release_D, level_Co, day_of_year, hour0)
        hp = SusquehannaModel.g_hydRevCo(
            np.asarray(release_D),
            level_Co,
            day_of_year,
            hour0,
            self.GG,
            self.gammaH20,
            self.tailwater,
            self.turbines,
            self.energy_prices,
        )
        # hp_mr = self.g_hydRevMR(q_pump, q_rel, level_Co, level_mr,
        # day_of_year, hour0)
        hp_mr = SusquehannaModel.g_hydRevMR(
            np.asarray(q_pump),
            np.asarray(q_rel),
            level_Co,
            np.asarray(level_mr),
            day_of_year,
            hour0,
            self.GG,
            self.gammaH20,
            self.turbines_Muddy,
            self.energy_prices,
        )
        # Revenue s_rr.extend([hp[1], hp_mr[2], hp_mr[3]])
        # Production s_rr.extend([hp[0], hp_mr[0], hp_mr[1]])
        return sto_co, sto_mr, rel_a, rel_b, rel_c, rel_d, hp[1], hp_mr[2], \
               hp_mr[3], hp[0], hp_mr[0], hp_mr[1]

    def g_storagereliability(self, h, h_target):
        c = 0
        Nw = 0

        # FIXME this probably can be fully vectorized
        # the modulus is not neeede dif h_target is just
        # expanded to match the length of h
        for i, h_i in np.ndenumerate(h):
            tt = i[0] % self.n_days_one_year
            if h_i < h_target[tt]:  # h[i] + 1  in flood model
                c = c + 1
            # if h_target[tt] > 0:
            #     Nw += 1

        G = 1 - c / np.sum(h_target > 0)
        return G

    def g_shortage_index(self, q1, qTarget):
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_days_one_year))
        maxarr = (qTarget * delta) - (q1 * delta)
        maxarr[maxarr < 0] = 0
        gg = maxarr / (qTarget * delta)
        g = np.mean(np.square(gg))
        return g

    def g_vol_rel(self, q1, qTarget):
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_days_one_year))
        g = (q1 * delta) / (qTarget * delta)
        G = utils.computeMean(g)
        return G

    def simulate(self, input_variable_list_var, inflow_MC_n_sim,
                 inflowLateral_MC_n_lat, inflow_Muddy_MC_n_mr,
                 evap_CO_MC_e_co, evap_Muddy_MC_e_mr, opt_met):
        # Initializing daily variables
        # storages and levels

        shape = (self.time_horizon_H + 1, )
        storage_co = np.empty(shape)
        level_co = np.empty(shape)
        storage_mr = np.empty(shape)
        level_mr = np.empty(shape)

        # Conowingo actual releases
        shape = (self.time_horizon_H,)
        release_a = np.empty(shape)
        release_b = np.empty(shape)
        release_c = np.empty(shape)
        release_d = np.empty(shape)

        # hydropower production/revenue
        hydropowerProduction_Co = []  # energy production at Conowingo
        hydropowerProduction_MR = []  # energy production at Muddy Run
        hydroPump_MR = []  # energy consumed for pumping at Muddy Run
        hydropowerRevenue_Co = []  # energy revenue at Conowingo
        hydropowerRevenue_MR = []  # energy revenue at Muddy Run
        hydropowerPumpRevenue_MR = []  # energy revenue consumed for pumpingat Muddy Run

        # release decision variables ( AtomicPP, Baltimore, Chester ) only
        # Downstream in Baseline
        uu = []

        theta = np.asarray(input_variable_list_var)
        center, radius, weights = rbf_functions.determine_parameters(theta,
            self.rbf_kwargs['n_inputs'], self.rbf_kwargs['n_rbf'])

        # initial condition
        level_co[0] = self.init_level
        storage_co[0] = self.level_to_storage(level_co[0], 1)
        level_mr[0] = self.init_level_MR
        storage_mr[0] = self.level_to_storage(level_mr[0], 0)

        # identification of the periodicity (365 x fdays)
        decision_steps_per_year = self.n_days_in_year * self.decisions_per_day
        year = 0

        # run simulation
        for t in range(self.time_horizon_H):
            day_of_week = (self.day0 + t) % 7
            day_of_year = t % self.n_days_in_year
            if day_of_year % self.n_days_in_year == 0 and t != 0:
                year = year + 1

            # subdaily variables
            shape = (self.decisions_per_day + 1,)
            daily_storage_co = np.empty(shape)
            daily_level_co = np.empty(shape)
            daily_storage_mr = np.empty(shape)
            daily_level_mr = np.empty(shape)

            shape = (self.decisions_per_day,)
            daily_release_a = np.empty(shape)
            daily_release_b = np.empty(shape)
            daily_release_c = np.empty(shape)
            daily_release_d = np.empty(shape)

            # initialization of sub-daily cycle
            daily_level_co[0] = level_co[t]  # level_co[day_of_year] <<< in flood
            daily_storage_co[0] = storage_co[t]
            daily_level_mr[0] = level_mr[t]
            daily_storage_mr[0] = storage_mr[t]

            # sub-daily cycle
            for j in range(self.decisions_per_day):
                decision_step = t * self.decisions_per_day + j

                # decision step i in a year
                jj = decision_step % decision_steps_per_year

                # compute decision
                if opt_met == 0:  # fixed release
                    # FIXME will crash because uu is empty list
                    uu.append(uu[0])
                elif opt_met == 1:  # RBF-PSO
                    rbf_input = np.asarray([jj, daily_level_co[j]])
                    uu = self.apply_rbf_policy(rbf_input, center, radius,
                                               weights)

                # system transition
                ss_rr_hp = self.res_transition_h(
                    daily_storage_co[j],
                    uu,
                    inflow_MC_n_sim[year][day_of_year],
                    inflowLateral_MC_n_lat[year][day_of_year],
                    evap_CO_MC_e_co[year][day_of_year],
                    daily_storage_mr[j],
                    inflow_Muddy_MC_n_mr[year][day_of_year],
                    evap_Muddy_MC_e_mr[year][day_of_year],
                    day_of_year,
                    day_of_week,
                    j,
                )

                daily_storage_co[j + 1] = ss_rr_hp[0]
                daily_storage_mr[j + 1] = ss_rr_hp[1]
                daily_level_co[j + 1] = self.storage_to_level(daily_storage_co[j + 1], 1)
                daily_level_mr[j + 1] = self.storage_to_level(daily_storage_mr[j + 1], 0)

                daily_release_a[j] = ss_rr_hp[2]
                daily_release_b[j] = ss_rr_hp[3]
                daily_release_c[j] = ss_rr_hp[4]
                daily_release_d[j] = ss_rr_hp[5]

                # Hydropower revenue production
                hydropowerRevenue_Co.append(
                    ss_rr_hp[6])  # 6-hours energy revenue ($/6h)
                hydropowerPumpRevenue_MR.append(
                    ss_rr_hp[7])  # 6-hours energy revenue ($/6h) at MR
                hydropowerRevenue_MR.append(
                    ss_rr_hp[8])  # 6-hours energy revenue ($/6h) at MR
                hydropowerProduction_Co.append(
                    ss_rr_hp[9])  # 6-hours energy production (kWh/6h)
                hydroPump_MR.append(
                    ss_rr_hp[10])  # 6-hours energy production (kWh/6h) at MR
                hydropowerProduction_MR.append(
                    ss_rr_hp[11])  # 6-hours energy production (kWh/6h) at MR


            # daily values
            level_co[day_of_year + 1] = daily_level_co[self.decisions_per_day]
            storage_co[t + 1] = daily_storage_co[self.decisions_per_day]
            release_a[day_of_year] = np.mean(daily_release_a)
            release_b[day_of_year] = np.mean(daily_release_b)
            release_c[day_of_year] = np.mean(daily_release_c)
            release_d[day_of_year] = np.mean(daily_release_d)
            level_mr[t + 1] = daily_level_mr[self.decisions_per_day]
            storage_mr[t + 1] = daily_storage_mr[self.decisions_per_day]

        # log level / release
        if self.log_objectives:
            self.blevel_CO.append(level_co)
            self.blevel_MR.append(level_mr)
            self.ratom.append(release_a)
            self.rbalt.append(release_b)
            self.rches.append(release_c)
            self.renv.append(release_d)

        # compute objectives
        # level_co.pop(0)
        j_hyd = sum(hydropowerRevenue_Co) / self.n_years / pow(10,
                                                              6)  # GWh/year (M$/year)
        j_atom = self.g_vol_rel(release_a, self.w_atomic)
        j_balt = self.g_vol_rel(release_b, self.w_baltimore)
        j_ches = self.g_vol_rel(release_c, self.w_chester)
        j_env = self.g_shortage_index(release_d, self.min_flow)
        j_rec = self.g_storagereliability(storage_co, self.h_ref_rec)

        return -j_hyd, -j_atom, -j_balt, -j_ches, j_env, -j_rec
