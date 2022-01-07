#include <opencv2/highgui.hpp>  // to display image.
#include <fstream>              // reading data from file.
#include <filesystem>           // setting-up the path of data file from current working directory.
#include <execution>            // concurrent execution of generating depth map from point cloud.

// Global variables.
// Intrinsic parameters
float fx{ 2172.1f }, fy{ 2171.6f }, ppx{ 960.8f }, ppy{ 602.5f };
// Image size.
int width{ 1440 }, height{ 900 };
// Depth range to display. 
float depthMin{ 1400.f }, depthMax{ 1600.f };

/**
* @brief Contains depth data given image coordinate.
*/
struct UVDepth
{
  int u, v;
  uchar depth;  // Type should be ushort or 10-bit. Using uchar is for convenience to display result.
  UVDepth(const int _u = 0, const int _v = 0, const uchar _depth = 0) : 
    u(_u), v(_v), depth(_depth) {}
};

/**
* @brief 3-d point.
*/
struct PointXYZ
{
  float x, y, z;
  PointXYZ(const float _x = 0.f, const float _y = 0.f, const float _z = 0.f) :
    x(_x), y(_y), z(_z) {}
};

/**
* @brief Generating depth map from point cloud data.
* 
* According to the instruction, we have
* (1) the camera frame origin is aligned to that of object frame.
* (2) the optical axis is aligned to z-axis of object frame, but the direction is opposite.
* In order to project points in object frame to camera, object point is expressed in camera frame as in,
* [x;y;z]_cam = Rctc * [x; y; z; 1]_obj
* From (1), tc = [0; 0; 0]
* From (2), Rc = [1, 0, 0; 0, -1, 0; 0, 0, -1]
* Rctc is the transformation of object frame w.r.t. camera.
* Then, points expressed in camera frame are projected using intrinsic matrix, K.
* lambda*[u; v; 1] = K * [x; y; z]_cam
* and, K = [fx, 0, ppx; 0, fy, ppy; 0, 0, 1] and lambda is a scale factor.
*
*/
void PointCloudToDepthMap(std::vector<PointXYZ>& _points)
{
  std::vector<UVDepth> uvDepth(_points.size());
  auto op = [](const PointXYZ& pt)  // capturing global intrinsic and depth range variables.
  {
    auto x{ pt.x }, y{ -pt.y }, z{ -pt.z };  // Equivalent to Rctc * [x; y; z; 1]_obj
    auto u{ static_cast<int>(std::round((fx * x + ppx * z) / z)) };  // lambda*[u; v; 1] = K * [x; y; z]_cam
    auto v{ static_cast<int>(std::round((fy * y + ppy * z) / z)) };  // The denominator is simplified according to the instruction.
    auto d{ std::clamp(pt.z, depthMin, depthMax) };  // Not mentioned in the instruction.
    d -= depthMin;                                   // Line 59, 60, 61: Depth normalization in order
    d *= UCHAR_MAX / (depthMax - depthMin);          // to display in 8-bit image.
    return UVDepth(u, v, UCHAR_MAX - static_cast<uchar>(d));  // Closer the depth, lighter the pixel.
  };

  // Transform input 3d point to a pixel in depth map.
  std::transform(std::execution::par, _points.begin(), _points.end(), uvDepth.begin(), op);

  // Diplay the result using OpenCV. The default pixel value is 0; no associated point.
  cv::Mat_<uchar> depthMap(height, width, uchar(0));
  
  for (auto it{ uvDepth.begin() }; it != uvDepth.end(); ++it)
    if (it->u >= 0 && it->v >= 0 && it->u < width && it->v < height)  // boundary check
      if (depthMap(it->v, it->u) < it->depth)  // handle occlusion
        depthMap(it->v, it->u) = it->depth;

  cv::imshow("depth", depthMap);
  cv::waitKey(0);
  cv::destroyWindow("depth");
}

int main()
{
  std::filesystem::path path = std::filesystem::current_path();
  path += std::filesystem::path("/../../data/DepthMap/pointcloud.txt");
  if (!std::filesystem::exists(path))
    return 1;

  std::fstream fs(path.string(), std::ios_base::in);

  std::vector<PointXYZ> points;
  points.reserve(1024 * 1024 * 16);

  float x, y, z;
  while (fs >> x >> y >> z)
    points.emplace_back(x, y, z);

  if (fs.is_open())
    fs.close();

  PointCloudToDepthMap(points);

  return 0;
}
