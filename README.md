This repository contains Python 3 implementations of some of the graph construstion methods I used for some of my experiments. 

## Installation
Clone the repository and open the folder on your terminal. Then: 

```bash
pip install -U .
```

Use `pip install -U .[cknn]` or `pip install -U .[distanceclosure]` if you want to use these methods as well.     

## Usage
```python
from GraphConstruction import methods
from GraphConstruction.methods import vectors2distance
D = vectors2distance(list_of_vectors, metric='cosine', normalised=False)
```

## References
This repository is completely experimental, thus there is no need to cite anything about it.
However, please cite the corresponding work when using any specific method through this collection. 
While the source code contains metadata including bibliographical and implementation sources for most of the methods, this metadata may not be complete. Thus, please take extra care and check for yourselves.
