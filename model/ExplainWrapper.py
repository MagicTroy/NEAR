import sys
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import LinearRegression


class ExplainWrapper:
    def __init__(self, embedding_size, kernel_width=None, kernel=None):
        self.embedding_size = embedding_size

        # if None, defaults to sqrt (embedding size) * 0.75
        # similarity kernel function, if None, defaults to be an exponential kernel
        if kernel_width is None:
            kernel_width = np.sqrt(self.embedding_size) * 0.9
            # kernel_width = 1. / self.embedding_size * 1.
        kernel_width = np.float32(kernel_width)

        self.kernel = lambda (distance): np.sqrt(
            np.exp(-(distance ** 2) / kernel_width ** 2)) if kernel is None else kernel

    def calculate_factor_weights(self, perturbs, the_wights):
        """
        :param perturbs: [None, embedding size, bins, embedding size]
        :param the_wights: [None, embedding size]
        :return:
        """
        factor_weights_list = []
        for _x, _y in zip(perturbs, the_wights):
            _x = _x.reshape(-1, self.embedding_size)

            distance = pairwise_distances(X=_x, Y=[_y])
            factor_weights = self.kernel(distance).flatten()
            factor_weights_list.append(factor_weights)

        # shape: [None, embedding size * bins]
        factor_weights_list = np.array(factor_weights_list)

        return factor_weights_list

    def explain_for_train(self, perturbs, perturb_y, factor_weights):
        """
        :param perturbs: [embedding size * bins, embedding size]
        :param perturb_y: [embedding size * bins]
        :param factor_weights: [embedding size * bins]
        :return:
        """
        easy_model = Ridge(alpha=1, fit_intercept=True, random_state=123)
        easy_model.fit(X=perturbs, y=perturb_y.reshape(-1, 1), sample_weight=factor_weights)

        # shape: [embedding size, bins]
        explain_predict = easy_model.predict(X=perturbs).reshape(self.embedding_size, -1)

        return explain_predict

    def explain_for_train_no_kernel(self, perturbs, perturb_y):
        """
        :param perturbs: [embedding size, bins, None, embedding size]
        :param perturb_y: [embedding size, bins, None]
        :return:
        """
        # FIXME: not working
        perturbs = perturbs.reshape(-1, self.embedding_size)
        perturb_y = perturb_y.flatten()

        # easy_model = Lasso(alpha=1., random_state=123)
        easy_model = Ridge(alpha=1, fit_intercept=True, random_state=123, max_iter=10000)
        # easy_model = LinearRegression(n_jobs=-1)
        easy_model.fit(X=perturbs, y=perturb_y.reshape(-1, 1))

        # shape: [embedding size * bins * None]
        explain_predict = easy_model.predict(X=perturbs)

        print(explain_predict)

        print(perturb_y)

        print(easy_model.score(X=perturbs, y=perturb_y.reshape(-1, 1)))

        sys.exit(3)
        return explain_predict


class PerturbEmbeddingExplain:
    def __init__(self, embedding_size, bins, k):
        self.embedding_size = embedding_size
        self.bins = bins
        self.k = k

    def find_factor_bins(self, data):
        # data shape: [num user/item, factor size]
        # return: {factor: factor bins (bins of lists)}
        factor_bins_dict = {}
        for idx, factor in enumerate(data.T):
            temp = np.sort(factor)
            patches = len(temp) // self.bins

            temp_list = []
            for num_bin in range(self.bins):
                start_index = num_bin * patches
                end_index = len(temp) if num_bin == self.bins - 1 else (num_bin + 1) * patches

                candidate_bins = temp[start_index: end_index]
                assert len(candidate_bins) > self.k, 'bins list is smaller than k, choose another k'

                temp_list.append(candidate_bins)
            factor_bins = np.array(temp_list)
            factor_bins_dict[idx] = factor_bins
        # {factor index: [[values] * bins]}
        return factor_bins_dict

    @staticmethod
    def create_perturb_embeddings(the_weights, the_factor_index, the_values):
        # the_weights = np.zeros(shape=the_weights.shape)
        the_weights[:, the_factor_index] = the_values
        return the_weights

    def bins_batch_perturbs(self, current_factor, factor_bins, user_weights, item_weights, user_bias, item_bias, model,
                            ratings, pre_or_loss='pred'):
        """
        :param current_factor: int
        :param factor_bins: [values]
        :param user_weights: [None, embedding size]
        :param item_weights: [None, embedding size]
        :param user_bias: [None]
        :param item_bias: [None]
        :param model: RS model
        :return:
        """
        bins_batch_perturbs = []
        bins_batch_perturb_predictions = []
        bins_batch_perturb_values = []
        for bins_index, bins in enumerate(factor_bins):
            # candidate factor value of this factor in a batch
            # shape: [None]
            candidate_factor_values = [value if value in bins else bins[np.random.choice(len(bins))] for value in
                                       user_weights[:, current_factor]]

            # update batch user factor values
            # shape: [None, embedding size]
            perturb_weights = self.create_perturb_embeddings(
                the_weights=user_weights, the_factor_index=current_factor, the_values=candidate_factor_values)

            if pre_or_loss == 'pred':
                # shape: [None]
                perturb_predict = model.predict_by_perturb_weights(
                    user_weights=perturb_weights, item_weights=item_weights,
                    user_bias=user_bias, item_bias=item_bias).flatten()
            else:
                # shape: [None]
                perturb_predict = model.loss_by_perturb_weights(
                    user_weights=perturb_weights, item_weights=item_weights,
                    user_bias=user_bias, item_bias=item_bias, rating=ratings).flatten()

            bins_batch_perturbs.append(perturb_weights)
            bins_batch_perturb_predictions.append(perturb_predict)
            bins_batch_perturb_values.append(candidate_factor_values)

        # shape: [bins, None, embedding size]
        bins_batch_perturbs = np.array(bins_batch_perturbs)
        # shape: [bins, None]
        bins_batch_perturb_predictions = np.array(bins_batch_perturb_predictions)
        # shape: [bins, None]
        bins_batch_perturb_values = np.array(bins_batch_perturb_values)

        # shape: [bins, None, embedding size]
        # shape: [bins, None]
        # shape: [bins, None]
        return bins_batch_perturbs, bins_batch_perturb_predictions, bins_batch_perturb_values

    def factor_bins_batch_perturbs(
            self, user_weights_bins_dict, user_weights, item_weights, user_bias, item_bias, model,
            ratings, pre_or_loss='pred'):
        """
        :param user_weights_bins_dict: dict
        :param user_weights: [None, embedding size]
        :param item_weights: [None, embedding size]
        :param user_bias: [None]
        :param item_bias: [None]
        :param model: RS model
        :return:
        """
        factor_bins_batch_perturbs = []
        factor_bins_batch_perturb_predictions = []
        factor_bins_batch_perturb_values = []
        for factor_index in range(self.embedding_size):
            # shape: [[values] * bins]
            factor_bins = user_weights_bins_dict.get(factor_index)

            # shape: [bins, None, embedding size]
            # shape: [bins, None]
            # shape: [bins, None]
            bins_batch_perturbs, bins_batch_perturb_predictions, bins_batch_perturb_values = self.bins_batch_perturbs(
                current_factor=factor_index, factor_bins=factor_bins,
                user_weights=user_weights, item_weights=item_weights, user_bias=user_bias, item_bias=item_bias,
                model=model, ratings=ratings, pre_or_loss=pre_or_loss)

            factor_bins_batch_perturbs.append(bins_batch_perturbs)
            factor_bins_batch_perturb_predictions.append(bins_batch_perturb_predictions)
            factor_bins_batch_perturb_values.append(bins_batch_perturb_values)

        # shape: [embedding size, bins, None, embedding size]
        factor_bins_batch_perturbs = np.array(factor_bins_batch_perturbs)
        # shape: [embedding size, bins, None]
        factor_bins_batch_perturb_predictions = np.array(factor_bins_batch_perturb_predictions)
        # shape: [embedding size, bins, None]
        factor_bins_batch_perturb_values = np.array(factor_bins_batch_perturb_values)

        return factor_bins_batch_perturbs, factor_bins_batch_perturb_predictions, factor_bins_batch_perturb_values

    def prepare_for_explain(self, model, user_weights, item_weights, user_bias, item_bias, ratings, pre_or_loss='pred'):
        """
        :param model:
        :param user_weights: original weights
        :param item_weights: original weights
        :param user_bias: original bias
        :param item_bias: original bias
        :return:
        """
        """ 1. get factor bins """
        # shape: {factor index: [[values] * bins]}
        user_weights_bins_dict = self.find_factor_bins(model.get_total_weights()[0])

        """ 2. create perturb weights """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.factor_bins_batch_perturbs(
            user_weights_bins_dict=user_weights_bins_dict,
            user_weights=user_weights, item_weights=item_weights, user_bias=user_bias, item_bias=item_bias, model=model,
            ratings=ratings,
            pre_or_loss=pre_or_loss)

        return batch_perturbs, batch_perturb_predictions, batch_perturb_values

    def perturb_item_prepare_for_explain(self, model, user_weights, item_weights, user_bias, item_bias, ratings):
        """
        :param model:
        :param user_weights: original weights
        :param item_weights: original weights
        :param user_bias: original bias
        :param item_bias: original bias
        :return:
        """
        """ 1. get factor bins """
        # shape: {factor index: [[values] * bins]}
        user_weights_bins_dict = self.find_factor_bins(model.get_total_weights()[1])

        """ 2. create perturb weights """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.factor_bins_batch_perturbs(
            user_weights_bins_dict=user_weights_bins_dict,
            user_weights=user_weights, item_weights=item_weights, user_bias=user_bias, item_bias=item_bias, model=model,
            ratings=ratings)

        return batch_perturbs, batch_perturb_predictions, batch_perturb_values

    def text_prepare_for_explain(self, model, user_weights, item_weights, user_bias, item_bias, ratings):
        """
        :param model:
        :param user_weights: original weights
        :param item_weights: original weights
        :param user_bias: original bias
        :param item_bias: original bias
        :return:
        """
        """ 1. get factor bins """
        # shape: {factor index: [[values] * bins]}
        user_weights_bins_dict = self.find_factor_bins(model.get_total_feature()[0])

        """ 2. create perturb weights """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.factor_bins_batch_perturbs(
            user_weights_bins_dict=user_weights_bins_dict,
            user_weights=user_weights, item_weights=item_weights, user_bias=user_bias, item_bias=item_bias, model=model,
            ratings=ratings)

        return batch_perturbs, batch_perturb_predictions, batch_perturb_values

    def explain_during_train(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_weights_bias(user=user_batch, item=item_batch)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ 3. calculate perturb factor weights """
        explainer = ExplainWrapper(embedding_size=self.embedding_size)

        """ get explain prediction """
        # shape: [embedding size * bins * None]
        explain_predict = explainer.explain_for_train_no_kernel(
            perturbs=batch_perturbs, perturb_y=batch_perturb_predictions)

        # shape: [embedding size, bins, None]
        batch_explain_predict = explain_predict.reshape(self.embedding_size, self.bins, -1)

        """ loss with actual ratings """
        for batch_index, actual_rating in enumerate(rating_batch):
            # shape: [embedding size, bins]
            explain_predict = batch_explain_predict[:, :, batch_index]
            square_losses = np.square(actual_rating - explain_predict)

            # print(square_losses)

            # shape: [embedding size]
            sorted_factor_index_by_min_bins_errors = np.argsort(np.min(square_losses, axis=1))
            selected_factor_index = np.random.choice(sorted_factor_index_by_min_bins_errors[:self.k])
            selected_bin_index = np.argmin(square_losses[selected_factor_index])

            selected_value = batch_perturb_values[:, :, batch_index][selected_factor_index][selected_bin_index]

            # print(np.min(square_losses, axis=1), np.argmin(square_losses, axis=1))
            # print(square_losses[selected_factor_index][selected_bin_index])

            user_weights[batch_index][selected_factor_index] = selected_value

            # print(square_losses[the_selected_factor_index_by_bins_min_loss, the_selected_bins_index])

            # sys.exit(3)

        return user_weights, item_weights, user_bias, item_bias

    def explain_during_train_by_listen(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_weights_bias(user=user_batch, item=item_batch)

        print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ loss """
        # shape: [embedding size, bins, None]
        batch_perturb_loss = np.square(rating_batch - batch_perturb_predictions)

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        user_bins_value_list = [batch_perturb_values[important_factor_index, :, user_index] for
                                user_index, important_factor_index in
                                zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None]
        user_selected_value = np.min(user_bins_value_list, axis=1)
        # shape: [None, bins] -> [None, bins] -> [None]
        # random_select_bin_index_by_k = np.random.choice(range(self.k), len(user_bins_value_list))
        # user_selected_value = np.argsort(user_bins_value_list, axis=1)[:, random_select_bin_index_by_k]

        for user_index, (factor_index, value) in enumerate(
                zip(user_most_important_factor_index_list, user_selected_value)):
            user_weights[user_index][factor_index] = value

            break

        return user_weights, item_weights, user_bias, item_bias

    def explain_during_train_by_listen_2(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_weights_bias(user=user_batch, item=item_batch)

        # print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ loss """
        # shape: [embedding size, bins, None]
        batch_perturb_loss = np.square(rating_batch - batch_perturb_predictions)

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        bin_batch_loss_list = [batch_perturb_loss[important_factor_index, :, user_index] for
                               user_index, important_factor_index in
                               zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None] bin index
        user_most_important_bins_index_list = np.argmin(bin_batch_loss_list, axis=1)

        temp = np.zeros(shape=user_weights.shape)

        for batch_index, (factor_index, bin_index) in enumerate(
                zip(user_most_important_factor_index_list, user_most_important_bins_index_list)):
            temp[batch_index] = batch_perturbs[factor_index, bin_index, batch_index, :]

        return temp, item_weights, user_bias, item_bias

    def explain_loss_during_train_by_listen_2(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_weights_bias(user=user_batch, item=item_batch)

        # print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch, pre_or_loss='pred')

        """ loss """
        # shape: [embedding size, bins, None]
        batch_perturb_loss = -(rating_batch * np.log(batch_perturb_predictions) + (1 - rating_batch) * np.log(1 - batch_perturb_predictions))
        # batch_perturb_loss = batch_perturb_predictions

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        bin_batch_loss_list = [batch_perturb_loss[important_factor_index, :, user_index] for
                               user_index, important_factor_index in
                               zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None] bin index
        user_most_important_bins_index_list = np.argmin(bin_batch_loss_list, axis=1)

        temp = np.zeros(shape=user_weights.shape)

        for batch_index, (factor_index, bin_index) in enumerate(
                zip(user_most_important_factor_index_list, user_most_important_bins_index_list)):
            temp[batch_index] = batch_perturbs[factor_index, bin_index, batch_index, :]

        return temp, item_weights, user_bias, item_bias

    def explain_during_train_by_listen_3(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_weights_bias(user=user_batch, item=item_batch)

        # print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.perturb_item_prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ loss """
        # shape: [embedding size, bins, None]
        batch_perturb_loss = np.square(rating_batch - batch_perturb_predictions)

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        bin_batch_loss_list = [batch_perturb_loss[important_factor_index, :, user_index] for
                               user_index, important_factor_index in
                               zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None] bin index
        user_most_important_bins_index_list = np.argmin(bin_batch_loss_list, axis=1)

        temp = np.zeros(shape=user_weights.shape)

        for batch_index, (factor_index, bin_index) in enumerate(
                zip(user_most_important_factor_index_list, user_most_important_bins_index_list)):
            temp[batch_index] = batch_perturbs[factor_index, bin_index, batch_index, :]

        return temp, item_weights, user_bias, item_bias

    def explain_text_during_train_by_listen(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_feature(user=user_batch, item=item_batch)

        # print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.text_prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ loss """
        # shape: [embedding size, bins, None]
        batch_perturb_loss = np.square(rating_batch - batch_perturb_predictions)

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        bin_batch_loss_list = [batch_perturb_loss[important_factor_index, :, user_index] for
                               user_index, important_factor_index in
                               zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None] bin index
        user_most_important_bins_index_list = np.argmin(bin_batch_loss_list, axis=1)

        temp = np.zeros(shape=user_weights.shape)

        for batch_index, (factor_index, bin_index) in enumerate(
                zip(user_most_important_factor_index_list, user_most_important_bins_index_list)):
            temp[batch_index] = batch_perturbs[factor_index, bin_index, batch_index, :]

        return temp, item_weights

    def explain_dmf_during_train_by_listen(self, model, user_batch, item_batch, rating_batch):
        """
        :param model:
        :param user_batch:
        :param item_batch:
        :param rating_batch:
        :return:
        """
        """ 1. get original weights and bias """
        # shape: [None, embedding size], [None, embedding size], [None], [None]
        user_weights, item_weights, user_bias, item_bias = model.get_feature(user=user_batch, item=item_batch)

        # print(user_weights.shape, item_weights.shape)

        """ 2. prepare for explain """
        # shape: [embedding size, bins, None, embedding size]
        # shape: [embedding size, bins, None]
        # shape: [embedding size, bins, None]
        batch_perturbs, batch_perturb_predictions, batch_perturb_values = self.text_prepare_for_explain(
            model=model, user_weights=user_weights, item_weights=item_weights,
            user_bias=user_bias, item_bias=item_bias, ratings=rating_batch)

        """ loss """
        # shape: [embedding size, bins, None]
        regRate = rating_batch / model.maxRate
        batch_perturb_loss = -(regRate * np.log(batch_perturb_predictions) + (1 - regRate) * np.log(
            1 - batch_perturb_predictions))

        # shape: [embedding size, None]
        factor_batch_loss_list = np.mean(batch_perturb_loss, axis=1)

        # shape: [None]
        user_most_important_factor_index_list = np.argmin(factor_batch_loss_list, axis=0)
        # user_most_important_factor_index_list = np.random.choice(np.argsort(factor_batch_loss_list, axis=0)[:self.k])

        # shape: [None, bins]
        bin_batch_loss_list = [batch_perturb_loss[important_factor_index, :, user_index] for
                               user_index, important_factor_index in
                               zip(range(len(user_batch)), user_most_important_factor_index_list)]

        # shape: [None] bin index
        user_most_important_bins_index_list = np.argmin(bin_batch_loss_list, axis=1)

        temp = np.zeros(shape=user_weights.shape)

        for batch_index, (factor_index, bin_index) in enumerate(
                zip(user_most_important_factor_index_list, user_most_important_bins_index_list)):
            temp[batch_index] = batch_perturbs[factor_index, bin_index, batch_index, :]

        return temp, item_weights
