from models.iw_vae import ImportanceWeightedVariationalAutoEncoder

class VariationalAutoEncoder(ImportanceWeightedVariationalAutoEncoder):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        number_of_monte_carlo_samples, analytical_kl_term = False,
        latent_distribution = "gaussian", number_of_latent_clusters = 1,
        parameterise_latent_posterior = False,
        reconstruction_distribution = None,
        number_of_reconstruction_classes = None,
        batch_normalisation = True, count_sum = True,
        number_of_warm_up_epochs = 0, epsilon = 1e-6,
        log_directory = "log",
        results_directory = results_directory):
        
        number_of_importance_samples = {
            "training": 1,
            "evaluation": 1
        }
        
        super(VariationalAutoEncoder, self).__init__(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples = number_of_monte_carlo_samples,
            number_of_importance_samples = number_of_importance_samples,
            analytical_kl_term = analytical_kl_term,
            latent_distribution = latent_distribution,
            number_of_latent_clusters = number_of_latent_clusters,
            parameterise_latent_posterior = parameterise_latent_posterior,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            epsilon = epsilon,
            log_directory = log_directory,
            results_directory = results_directory)
        
        self.type = "VAE"
