# Reducing the dimensionality
def rearranege_reduction(I, r1, r2, batch):

    # bsize, n*r1, m*r2, 1
    bsize, a, b, c = I.get_shape().as_list()
    bsize = batch
    n = a/r1
    m = b/r2
    X = tf.reshape(I,(bsize, a, m, r2, c)) # bsize, n*r1, m, r2, c
    X = tf.transpose(X,(0,1,2,4,3)) # bsize, n*r1, m , c , r2
    X = tf.split(X, r2, 4) # r2, [bsize, n*r1, m, c]
    X = tf.concat([x for x in X], 3) # bsize, n*r1, m , c*r2
    X = tf.reshape(X, (bsize, a, m, c*r2))
    X = tf.reshape(X,(bsize, n, r1, m, c*r2)) # bsize, n, r1, m, c*r2
    X = tf.transpose(X,(0,1,3,4,2)) # bsize, n, m , c*r2 , r1
    X = tf.split(X, r1, 4) # r1, [bsize, n, m, c*r2]
    X = tf.concat([x for x in X], 3) # bsize, n, m , c*r1*r2
    X = tf.reshape(X, (bsize, n, m, c*r1*r2))

    return X

# Increasing the dimensionality
def rearranege_boosting(I, r1, r2, batch):

    # bsize, a, b, r1*r2
    bsize, a, b, c = I.get_shape().as_list()
    bsize = batch
    X = tf.reshape(I, (bsize, a, b, r1, r2))
    X = tf.split(X, a, 1)  # a, [bsize, b, r1, r2]
    X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, b, a*r1, r2
    X = tf.split(X, b, 1)  # b, [bsize, a*r1, r2]
    X = tf.concat([tf.squeeze(x) for x in X],2)  #bsize, a*r1, b*r2
    return tf.reshape(X, (bsize, a*r1, b*r2, 1))
