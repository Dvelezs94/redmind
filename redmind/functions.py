#################################
# Learning rate decay functions #
#################################
def lr_decay(learning_rate, epoch, decay_rate):
    """Standard learning rate decay algorithm"""
    return (1 / (1 + decay_rate * epoch)) * learning_rate