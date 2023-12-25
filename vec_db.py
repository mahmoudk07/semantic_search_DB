import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
BITS2DTYPE = {
    ## our nbits
    8: np.uint8,
    9: np.uint16,
    10:np.uint16,  
    11:np.uint16,
    12:np.uint16,
    13:np.uint16,
    14:np.uint16,
    15:np.uint16,
    16:np.uint16,
    17:np.uint32,
    18:np.uint32,
    19:np.uint32,
    20:np.uint32,
    21:np.uint32,
    22:np.uint32,
    23:np.uint32,
    24:np.uint32,
}


class VecDB:
    """Custom IndexPQ implementation.

    Parameters
    ----------
    d
        Dimensionality of the original vectors.

    m
        Number of segments.

    nbits
        Number of bits.

    estimator_kwargs
        Additional hyperparameters passed onto the sklearn KMeans
        class.

    """

    def __init__(
        self,
        file_path:str="PATH_DB_10K", 
        new_db:bool=True,
        d:int=70,
        nbits:int=13,
        m:int=14,
         
        
    ) -> None:
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")

        if nbits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")

        self.m = m       
        self.k = 2**nbits
        self.d = d
        self.ds = d // m  ## 3dd variables fe subvector 
        self.file_path=file_path
        self.new_db=new_db
        self.estimators = [
            KMeans(n_clusters=self.k, init='random',max_iter=20) for _ in range(m) ## Kmeans lkol subvector
        ]

        self.is_trained = False

        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32
        ## codes table needed to be stored on disk
        self.codes: np.ndarray | None = None

    def train(self, X: np.ndarray) -> None:
        """Train all KMeans estimators.
         neede to be changed with large numbers
        Parameters
        ----------
        X
            Array of shape `(n, d)` and dtype `float32`.

        """
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed")

        for i in range(self.m):
            estimator = self.estimators[i]  ## bngeeb kol kmean
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            
            estimator.fit(X_i)

        self.is_trained = True


    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode original features into codes.
           run after train to get code table 
        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        Returns
        -------
        result
            Array of shape `(n_queries, m)` of dtype `np.uint8` or np.uint16 or np.uint32 .
        """
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)

        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            result[:, i] = estimator.predict(X_i)
        np.save('resultdata.npy', result)
        return result     ### codes table

    def add(self, X: np.ndarray) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        X
            Array of shape `(n_codes, d)` of dtype `np.float32`.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        self.codes = self.encode(X)    ### save table 

    def compute_asymmetric_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute asymmetric distances to all database codes.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        Returns
        -------
        distances
            Array of shape `(n_queries, n_codes)` of dtype `np.float32`.

        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")

        if self.codes is None:
            raise ValueError("No codes detected. You need to run `add` first")

        n_queries = len(X)
        n_codes = len(self.codes)

        distance_table = np.empty(
            (n_queries, self.m, self.k), dtype=self.dtype_orig
        )  # (n_queries, m, k)

        for i in range(self.m):
            X_i = X[:, i * self.ds : (i + 1) * self.ds]  # (n_queries, ds)
            centers = self.estimators[i].cluster_centers_  # (k, ds)
            #distance_table[:, i, :] = euclidean_distances(X_i, centers, squared=True)
            distance_table[:, i, :] = np.sqrt(np.sum(np.square(euclidean_distances(X_i, centers)), axis=-1))

        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)

        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]

        return distances
    def compute_asymmetric_distances_2(self, X: np.ndarray) -> np.ndarray:
        """Compute asymmetric distances to all database codes.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        Returns
        -------
        distances
            Array of shape `(n_queries, n_codes)` of dtype `np.float32`.

        """
       

        n_queries = len(X)


        distance_table = np.empty(
            (n_queries, 10, 2**16), dtype=self.dtype_orig
        )  # (n_queries, m, k)
        center=np.load(self.file_path+'/centroids.npy').astype(np.float32)
        for i in range(10):
            X_i = X[:, i * 7 : (i + 1) * 7]  # (n_queries, ds)
            centers = center[i]  # (k, ds)
            distance_table[:, i, :] = euclidean_distances(
                X_i, centers, squared=True
            )
        del center, centers
        codes=np.load(self.file_path+'/codes.npy').astype(np.uint16)
        n_codes = len(codes)

        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)

        for i in range(10):
            distances += distance_table[:, i, codes[:, i]]

        return distances
            

    def retrive(self, X: np.ndarray, k: int) -> tuple:
        """Find k closest database codes to given queries.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        k
            The number of closest codes to look for.

        Returns
        -------
        distances
            Array of shape `(n_queries, k)`.

        indices
            Array of shape `(n_queries, k)`.
        """

        n_queries = len(X)
        if(self.new_db):
            distances_all = self.compute_asymmetric_distances(X)
        else:
            distances_all = self.compute_asymmetric_distances_2(X)
        
        indices = np.argsort(distances_all, axis=1)[:, :k]
        flattened_indices = [index for sublist in indices for index in sublist]
        return flattened_indices
            
        """
            distances = np.empty((n_queries, k), dtype=np.float32)
            for i in range(n_queries):
                distances[i] = distances_all[i][indices[i]]
        """
        
        
            
    
    
    
    def insert_records(self,records_dict):
        extracted_vectors = np.array([record["embed"] for record in records_dict], dtype=np.float32)
        self.train( extracted_vectors)
        self.add(extracted_vectors)

        
        
        
        

"""   
## database load & query load    
query_vector =  np.random.rand(1, 70).astype(np.float32)
data = np.load('vectorsdata.npy')

## distances from query real approach 
distances = np.linalg.norm(data[:, :] - query_vector, axis=1)

## nearest indices from query real approach 
sorted_indices = np.argsort(distances)

vectors = data[sorted_indices]
sorted_dist=distances[sorted_indices] 

data = [{"id": i, "embed": list(row)} for i, row in enumerate(data)]
## initialize our pQ
index= CustomIndexPQ(file_path="15milion",new_db=False)

#index.insert_records(data)
## Train our pQ


## Add data to pq to build table
start_time = time.time()
indices_PQ= index.retrive(query_vector , 10)
end_time = time.time()
print(f"elapsed time: {end_time - start_time}")
real_indices=sorted_indices[0:30]
is_in_sorted = np.isin(indices_PQ, real_indices)

count_found = np.count_nonzero(is_in_sorted)
print(f"Values found in sorted_indices: {count_found}")

"""




##code to generate and  save to file
"""
import numpy as np
np.random.seed(50)
vectors = np.random.rand(15000000, 70).astype(np.float32)
np.save('vectorsdata.npy', vectors)
"""

