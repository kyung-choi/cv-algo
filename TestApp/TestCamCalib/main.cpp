#include "dlt.h"
#include "tsai.h"
#include <iostream>
#include <fstream>
#include <filesystem>

std::vector<float> loadData(const std::filesystem::path& _file);
std::vector<cv::Point2f> loadImgPts(const std::filesystem::path& _dir);
std::vector<cv::Point3f> loadObjPts(const std::filesystem::path& _dir);

int main()
{
  std::filesystem::path dir{ std::filesystem::current_path() };
  dir += std::filesystem::path("/../../data/CamCalib/");

  if (!std::filesystem::exists(dir))
    return 1;

  // Populate correspondences from files.
  std::vector<cv::Point2f> uv{ loadImgPts(dir) };
  std::vector<cv::Point3f> xyz{ loadObjPts(dir) };

  // Camera calibration
  bool DLT{ true };
  std::unique_ptr<ky::CamCalib> calibrator{ nullptr };

  try
  {
    if (DLT)
      calibrator = std::make_unique<ky::DLT>(uv, xyz);
    else
      calibrator = std::make_unique<ky::Tsai>(uv, xyz, cv::Size(1024, 1024));

    if (calibrator->Calibrate())
    {
      auto rms2d{ calibrator->RMS() };    // RMS of reprojection error.
      auto rms3d{ calibrator->RMS3D() };  // RMS of point to ray distance.
      auto K{ calibrator->K() };          // Camera matrix.
    }
  }
  catch (const cv::Exception& ce)
  {
    std::cout << ce.what();
  }
  catch (const std::runtime_error& re)
  {
    std::cout << re.what();
  }
  catch (const std::exception& ex)
  {
    std::cout << ex.what();
  }

  return 0;
}

std::vector<float> loadData(const std::filesystem::path& _file)
{
  if (!std::filesystem::exists(_file))
    return std::vector<float>();

  std::vector<float> data;
  float val;
  std::fstream fs(_file.c_str(), std::ios_base::in);
  while (fs >> val)
    data.push_back(val);

  fs.close();

  return data;
}

std::vector<cv::Point2f> loadImgPts(const std::filesystem::path& _dir)
{
  std::filesystem::path path(_dir);
  path += std::filesystem::path("uv.txt");

  std::vector<float> data{ loadData(path) };
  assert(!data.empty() && data.size() % 2 == 0);
  std::vector<cv::Point2f> pts(data.size() / 2);

  auto dataIter{ data.begin() };
  for (auto ptsIter{ pts.begin() }; ptsIter != pts.end(); ++ptsIter)
  {
    ptsIter->x = *dataIter++;
    ptsIter->y = *dataIter++;
  }

  return pts;
}

std::vector<cv::Point3f> loadObjPts(const std::filesystem::path& _dir)
{
  std::filesystem::path path(_dir);
  path += std::filesystem::path("xyz.txt");

  std::vector<float> data{ loadData(path) };
  assert(!data.empty() && data.size() % 3 == 0);
  std::vector<cv::Point3f> pts(data.size() / 3);

  auto dataIter{ data.begin() };
  for (auto ptsIter{ pts.begin() }; ptsIter != pts.end(); ++ptsIter)
  {
    ptsIter->x = *dataIter++;
    ptsIter->y = *dataIter++;
    ptsIter->z = *dataIter++;
  }

  return pts;
}
