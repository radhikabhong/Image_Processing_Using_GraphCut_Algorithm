# Graphcut_Algorithm

## Description
I have used the algorithm discussed in paper "Graphcut Textures: Image and Video Synthesis Using Graph Cuts" to synthesize an image with significantly larger dimensions from a sample image.

## Step By Step Process
The overall flow of the process goes as follows:
* First select the sample patch image that will be  randomly positioned in final large dimension image. 
* For overlapped images we need to implement the graphcut algorithm on the overlapped pixels for smoothening the edges of the patch.
* We represent this overlapped region in the form of graph and run edmonds karp minimum cut algorithm over it to ensure these smooth edges and this minimum cut is treated as the boundary for the inclusion of pixels from the overlapped images.
* This process is repeated until we reach the maximum dimensions of the output image.

## Project Implementation
### Ford-Fulkerson Algorithm
The basic working of the Ford-Fulkerson algorithm goes as follows:
1. Construct the residual graph for given graph.
2. Attempt to find a path from source to sink in the residual graph.
3. If a path exists, it gives us an augmenting path, which we use to improve given graph and go back to step 1.
4. If no path exists, we obtain minimum cut whose capacity is equal to the value of given graph. We know given graph is optimal, and the cut gives us a certificate of optimality.

The running time of the Ford-Fulkerson Algorithm is O(∣E∣⋅f)
where |E| is the number of edges and f is the maximum flow.

### Edmonds-Karp Algorithm
The Edmonds-Karp Algorithm is a specific implementation of the Ford-Fulkerson method.
1. In particular, Edmonds-Karp algorithm implements the searching for an augmenting path using the Breadth First Search (BFS) algorithm to find the shortest path from source to sink and the minimum residual capacity along that path, min. 
2. We then walk the augmenting path from target to source. 
3. Using the minium residual capacity, we reduce all residual capacities on the augmenting path by min and increase the residual capacities on the reverse edges (representing the flow).

The Edmonds-Karp algorithm runs in O(VE2)

## Statement of Help
How to install dependencies : `pip install -r requirements.txt`

How to run the code : `python3 .\code\graphcut.py $sample_img_name_path$ $output_img_name_path$ $output_height$ $output_width$`
eg. `python3 .\code\graphcut.py .\images\input\green.gif result.jpeg 512 512`

How to interpret the output : The resulting image will be generated in result.jpeg after code execution. This resulting output is a synthesized image from the sample image. The dimensions of this image will be equal to the dimensions specified above (512 * 512).

## Implementation 
### Libraries Used
* cv2 -> To perform read and write operations using images.
* numpy -> To effectively deal with 2d and 3d matrices.
* matplotlib -> To plot graphs.
* networkx -> To create graphs from the overlapped pixels.

## Input / Output
<pre>     <img src="strawberry.jpg" width="150" height="150">  ---------------------------------->   <img src="final_strawberry.jpg" width="400" height="400"> </pre>
<pre>     <img src="green.jpg" width="150" height="150">  ---------------------------------->   <img src="final_green.jpg" width="400" height="400"> </pre>


## Converting problem/solution
<pre>    <img src="graph_conversion.png" width="600" height="400">        </pre>

<pre>    <img src="circular_img_plot_1.png" width="600" height="400">     </pre>
<pre>    <img src="circular_img_plot_2.png" width="600" height="400">     </pre>
<pre>    <img src="circular_img_plot_4.png" width="600" height="400">     </pre>
<pre>    <img src="circular_img_plot_7.png" width="600" height="400">     </pre>
<pre>    <img src="Image_cut_1.png" width="600" height="400">             </pre>
<pre>    <img src="Image_cut_2.png" width="600" height="400">             </pre>
<pre>    <img src="Image_cut_3.png" width="600" height="400">             </pre>
<pre>    <img src="Image_cut_4.png" width="600" height="400">                 </pre>
<pre>    <img src="Image_cut_5.png" width="600" height="400">                 </pre>
<pre>    <img src="Image_cut_6.png" width="600" height="400">                 </pre>







