"""Offline audit of insertion logs 07/08; no robot clients or motion."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('insertion_audit', ROOT / 'logs/jaka_joy_trial_1013_01/analysis/insertion_diagnosis.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix


def load(trial):
    base.SESSION = ROOT / 'logs' / f'jaka_joy_trial_1013_{trial:02d}'
    d = base.load()
    d['trial'] = trial
    return d



def rotation_application_error(d):
    ix=np.flatnonzero(d['active'])
    maximum=0.0
    for k in ix:
        row=d['rows'][k]
        vector=d['R'][k]@np.array(row['admittance_integrated']['offset'][3:])
        angle=np.linalg.norm(vector)
        K=np.array([[0.,-vector[2],vector[1]],[vector[2],0.,-vector[0]],[-vector[1],vector[0],0.]])
        a=np.sin(angle)/angle if angle>1e-8 else 1.-angle*angle/6.
        b=(1.-np.cos(angle))/(angle*angle) if angle>1e-8 else .5-angle*angle/24.
        delta=np.eye(3)+a*K+b*(K@K)
        nominal=rotation_matrix(row['admittance_integrated']['nominal_pose']['quaternion_wxyz'])
        actual_reference=rotation_matrix(row['corrected_pose']['quaternion_wxyz'])
        maximum=max(maximum,float(np.max(np.abs(delta@nominal-actual_reference))))
    return maximum


def overview(d):
    t, R, p = d['t'], d['R'], d['p']
    relative_position = (p-p[0]) @ R[0] * 1000
    angles = np.rad2deg(np.array([rotation_vector_from_matrix(R[0].T @ r) for r in R]))
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, constrained_layout=True)
    labels = ('X', 'Y', 'Z')
    offset = d['offset'].copy(); offset[~d['active']] = np.nan
    jog = np.array([r['jog']['linear_m_s'] for r in d['rows']])
    angular = np.rad2deg(np.array([r['jog']['angular_rad_s'] for r in d['rows']]))
    forward = np.einsum('ni,ni->n', jog, R[:, :, 2])*1000
    groups = [(axes[0, 0], relative_position, 'actual TCP in initial tool axes (mm)'),
              (axes[0, 1], angles, 'actual orientation in initial tool axes (deg)'),
              (axes[1, 0], d['processed'][:, :3], 'processed tool force (N)'),
              (axes[1, 1], d['processed'][:, 3:], 'processed tool torque (Nm)'),
              (axes[2, 0], offset[:, :3]*1000, 'admittance translation offset (mm)'),
              (axes[2, 1], np.rad2deg(offset[:, 3:]), 'admittance rotation offset (deg)'),
              (axes[3, 1], angular, 'manual angular jog (deg/s)')]
    for ax, data, label in groups:
        for i in range(3): ax.plot(t, data[:, i], label=labels[i], lw=.8)
        ax.set_ylabel(label)
    axes[3, 0].plot(t, forward, label='forward jog along tool Z')
    axes[3, 0].set_ylabel('manual forward velocity (mm/s)')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(loc='upper left', ncol=3)
    for ax in axes[-1]: ax.set_xlabel('elapsed s')
    fig.suptitle(f'Trial {d["trial"]:02d}: full-session overview')
    fig.savefig(OUT / f'trial{d["trial"]:02d}_overview.png', dpi=125)
    plt.close(fig)



def contact_window(d, lo, hi):
    ix=np.flatnonzero((d['t']>=lo)&(d['t']<hi)&d['active'])
    t=d['t'][ix]; rel=t-t[0]; dt=float(np.median(np.diff(t)))
    line=np.c_[np.ones(len(t)),rel]
    R0=d['R'][ix[0]]
    signals={}
    rotations={}
    for key, poskey in [('actual','p'),('reference','reference'),('command','command'),('nominal','nominal')]:
        if key=='actual': rotations[key]=d['R'][ix]
        else:
            field={'reference':'corrected_pose','command':'final_command_tcp_pose'}
            if key=='nominal': rr=[d['rows'][k]['admittance_integrated']['nominal_pose'] for k in ix]
            else: rr=[d['rows'][k][field[key]] for k in ix]
            rotations[key]=np.array([rotation_matrix(r['quaternion_wxyz']) for r in rr])
        angle=np.rad2deg(np.array([rotation_vector_from_matrix(R0.T@r) for r in rotations[key]]))
        signals[key]=np.c_[(d[poskey][ix]-d['p'][ix[0]])@R0*1000,angle]
    detrended=signals['actual'][:,:3]-line@np.linalg.lstsq(line,signals['actual'][:,:3],rcond=None)[0]
    freq=np.fft.rfftfreq(len(t),dt)
    spectrum=np.abs(np.fft.rfft(detrended*np.hanning(len(t))[:,None],axis=0))
    band=(freq>=.4)&(freq<=4.)
    dominant_axis=int(np.argmax(np.max(spectrum[band,:2],axis=0)))
    peak=float(freq[band][np.argmax(spectrum[band,dominant_axis])])
    resolution=1/(t[-1]-t[0])
    candidates=np.linspace(max(.4,peak-resolution),min(4.,peak+resolution),301)
    def basis(f): return np.c_[line,np.sin(2*np.pi*f*rel),np.cos(2*np.pi*f*rel)]
    errors=[np.mean((detrended[:,dominant_axis]-basis(f)@np.linalg.lstsq(basis(f),detrended[:,dominant_axis],rcond=None)[0])**2) for f in candidates]
    peak=float(candidates[np.argmin(errors)])
    B=basis(peak)
    def harmonic(a):
        coef=np.linalg.lstsq(B,a,rcond=None)[0]
        return dict(peak_to_peak=(2*np.hypot(coef[-2],coef[-1])).tolist(),phase=np.arctan2(coef[-1],coef[-2]).tolist())
    hh={key:harmonic(value) for key,value in signals.items()}
    force_initial=np.einsum('nij,nj->ni',d['R'][ix],d['external'][ix,:3])@R0
    hf=harmonic(force_initial)
    force_phase=np.array(hf['phase'])
    actual_phase=np.array(hh['actual']['phase'][:3])
    phase_difference=np.rad2deg((force_phase-actual_phase+np.pi)%(2*np.pi)-np.pi)
    apparent_stiffness=np.array(hf['peak_to_peak'])/np.maximum(np.array(hh['actual']['peak_to_peak'][:3]),1e-12)
    sensor_position=d['p'][ix]+np.einsum('nij,j->ni',d['R'][ix],d['m']['tool_to_sensor_m'])
    sensor_relative=(sensor_position-sensor_position[0])@R0*1000
    qcmd=np.array([d['rows'][k]['final_command_q_rad'] for k in ix]); qact=np.array([d['rows'][k]['measured_q_rad'] for k in ix])
    qcmd-=line@np.linalg.lstsq(line,qcmd,rcond=None)[0];qact-=line@np.linalg.lstsq(line,qact,rcond=None)[0]
    lags=[]
    for axis in range(6):
        if np.std(qcmd[:,axis])<1e-5: lags.append(None);continue
        cc=[np.corrcoef(qcmd[:-k,axis] if k else qcmd[:,axis],qact[k:,axis] if k else qact[:,axis])[0,1] for k in range(45)]
        k=int(np.nanargmax(cc));lags.append(dict(ms=k*dt*1000,correlation=float(cc[k])))
    rows=[d['rows'][k] for k in ix]
    vel=np.array([r['admittance_integrated']['velocity'] for r in rows])
    x=np.array([r['admittance_integrated']['offset'] for r in rows])
    f=d['processed'][ix];mask=f!=0
    command_rot_error=np.array([np.linalg.norm(rotation_vector_from_matrix(a@b.T)) for a,b in zip(rotations['reference'],rotations['command'])])
    nominal_jog=np.array([r['jog']['linear_m_s'] for r in rows])
    forward=np.einsum('ni,ni->n',nominal_jog,d['R'][ix,:,2])*1000
    calibration=d['m']['runtime_calibration'];gc=calibration['gravity_compensation_config'];rts=np.array(d['m']['sensor_to_tool_rotation'])
    rbs=d['R'][ix]@rts;gravity=gc['payload_mass_kg']*np.array(gc['gravity_m_s2'])
    fg=np.einsum('nij,i->nj',rbs,gravity)
    direct=d['comp'][ix,3:]@rts.T;lever=np.cross(np.array(d['m']['tool_to_sensor_m']),d['external'][ix,:3])
    # Exact translational offset decomposition in base coordinates.
    R=d['R'][ix];rmid=(R[1:]+R[:-1])/2;xmid=(x[1:,:3]+x[:-1,:3])/2
    dRmotion=np.einsum('nij,nj->ni',np.diff(R,axis=0),xmid)*1000
    anglespan=np.ptp(signals['actual'][:,3:],axis=0)
    result=dict(window_s=[float(t[0]),float(t[-1])],rows=len(ix),dominant_initial_tool_axis='XY'[dominant_axis],
        dominant_frequency_hz=peak,harmonic=hh,
        contact_force_initial_axes_harmonic=hf,
        force_position_phase_deg=phase_difference.tolist(),
        apparent_force_position_amplitude_ratio_n_mm=apparent_stiffness.tolist(),
        sensor_position_harmonic=harmonic(sensor_relative),
        sensor_position_change_mm=(sensor_relative[-1]-sensor_relative[0]).tolist(),
        force_harmonic=harmonic(d['external'][ix,:3]),
        direct_torque_harmonic=harmonic(direct),lever_torque_harmonic=harmonic(lever),
        raw_sensor_force_harmonic=harmonic(d['raw'][ix,:3]),predicted_gravity_sensor_force_harmonic=harmonic(fg),
        command_to_actual_lags=lags,force_mean_n=f[:,:3].mean(0).tolist(),force_peak_norm_n=float(np.linalg.norm(f[:,:3],axis=1).max()),
        processed_torque_peak_abs_nm=np.max(abs(f[:,3:]),axis=0).tolist(),
        processed_wrench_nonzero_percent=(mask.mean(0)*100).tolist(),
        deadband_switches=np.sum(mask[1:]!=mask[:-1],axis=0).tolist(),
        velocity_saturation_percent=(np.mean(abs(vel)>=.004999,axis=0)*100).tolist(),
        angular_jog_peak_deg_s=np.rad2deg(np.max(abs(np.array([r['jog']['angular_rad_s'] for r in rows])),axis=0)).tolist(),
        forward_jog_mean_mm_s=float(forward.mean()),
        actual_pose_change_mm_deg=(signals['actual'][-1]-signals['actual'][0]).tolist(),
        nominal_pose_change_mm_deg=(signals['nominal'][-1]-signals['nominal'][0]).tolist(),
        actual_rotation_span_deg=anglespan.tolist(),
        admittance_offset_start_end_mm_deg=np.c_[x[[0,-1],:3]*1000,np.rad2deg(x[[0,-1],3:])].tolist(),
        reference_command_position_rms_mm=float(np.sqrt(np.mean(np.sum((d['reference'][ix]-d['command'][ix])**2,axis=1))))*1000,
        reference_command_rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(command_rot_error**2)))),
        rotation_of_translation_offset_path_mm=float(np.sum(np.linalg.norm(dRmotion,axis=1))))
    return result



def comparison_plot(data):
    fig, axes=plt.subplots(5,2,figsize=(14,13),constrained_layout=True)
    for col,(d,lo,hi,axis) in enumerate(((data[7],64,70.5,1),(data[8],80,86,0))):
        ix=np.flatnonzero((d['t']>=lo)&(d['t']<hi)&d['active'])
        t=d['t'][ix];rel=t-t[0];R0=d['R'][ix[0]];line=np.c_[np.ones(len(t)),rel]
        for key,label in [('reference','admittance reference'),('command','final command'),('p','joint-feedback TCP')]:
            y=((d[key][ix]-d['p'][ix[0]])@R0)[:,axis]*1000
            y-=line@np.linalg.lstsq(line,y,rcond=None)[0]
            axes[0,col].plot(t,y,label=label,lw=1)
        for key,label in [('filtered','filtered'),('processed','after deadband')]:
            axes[1,col].plot(t,d[key][ix,axis],label=label,lw=1)
        for limit in (-.5,.5): axes[1,col].axhline(limit,color='gray',ls=':',lw=.8)
        velocity=np.array([d['rows'][k]['admittance_integrated']['velocity'] for k in ix])[:,axis]*1000
        axes[2,col].plot(t,velocity,label='lateral admittance velocity')
        for limit in (-5,5): axes[2,col].axhline(limit,color='gray',ls=':',lw=.8)
        for j,label in enumerate(('Tx','Ty','Tz')): axes[3,col].plot(t,d['processed'][ix,3+j],label=label,lw=1)
        angles=np.rad2deg(np.array([rotation_vector_from_matrix(R0.T@r) for r in d['R'][ix]]))
        for j,label in enumerate(('Rx pitch','Ry','Rz')):axes[4,col].plot(t,angles[:,j],label=label,lw=1)
        axes[0,col].set_title(f'Trial {d["trial"]:02d}: '+('insertion, tool Y lateral' if col==0 else 'retraction, tool X lateral'))
        units=['detrended lateral TCP (mm)','lateral force (N)','admittance velocity (mm/s)','processed torque (Nm)','actual relative rotation (deg)']
        for row,label in enumerate(units):
            axes[row,col].set_ylabel(label);axes[row,col].legend(fontsize=8,loc='upper left');axes[row,col].grid(alpha=.2)
        axes[-1,col].set_xlabel('elapsed s')
    fig.savefig(OUT/'contact_oscillation.png',dpi=150)
    plt.close(fig)


def main():
    results = {}
    data = {}
    for trial in (7, 8):
        d = load(trial)
        data[trial] = d
        results[str(trial)] = dict(audit=base.helpers.audit(d), geometry=base.helpers.geometry(d),
                                  config=d['m']['effective_args'])
        results[str(trial)]['audit']['rotation_offset_application_max_abs']=rotation_application_error(d)
        overview(d)
        windows = [(30,38),(50,60),(64,70.5)] if trial==7 else [(50,53.8),(76.5,80),(80,86)]
        results[str(trial)]['contact_windows']=[contact_window(d,*w) for w in windows]
        for w in results[str(trial)]['contact_windows']:
            print('CONTACT', json.dumps(w))
        print('trial', trial)
        for lo in np.arange(0, d['t'][-1], 5):
            ix = np.flatnonzero((d['t']>=lo)&(d['t']<lo+5)&d['active'])
            if len(ix)<20: continue
            rr=[d['rows'][k] for k in ix]
            jog=np.array([r['jog']['linear_m_s'] for r in rr])
            forward=np.einsum('ni,ni->n',jog,d['R'][ix,:,2])*1000
            angular=np.rad2deg(np.array([r['jog']['angular_rad_s'] for r in rr]))
            print(f'{lo:5.1f}-{lo+5:5.1f}', 'n',len(ix),'forward',np.round(forward.mean(),2),
                  'Fmean',np.round(d['processed'][ix,:3].mean(0),2),
                  'Tstd',np.round(d['processed'][ix,3:].std(0),2),
                  'manualRotMax',np.round(np.max(abs(angular),axis=0),2))
    comparison_plot(data)
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
