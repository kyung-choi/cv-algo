#include "pose_est.h"

namespace ky
{
	PoseEst::PoseEst(const float _fx, const float _fy, const float _ppx, const float _ppy) :
		m_fx(_fx), m_fy(_fy), m_ppx(_ppx), m_ppy(_ppy) {}
}