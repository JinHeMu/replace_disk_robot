"""Independent URDF FK and logged force/admittance direction audit; no SDK."""
from pathlib import Path
import sys,json,csv,hashlib,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
URDF_RELATIVE='simulation/mujoco/models/jaka/urdf/tracer_jaka_zu5.urdf'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tool'))
from replace_disk_robot.core.rotation import rotation_matrix


def rpy(v):
    x,y,z=v;cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1.,0],[-sy,0,cy]])@np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])


def fk(xml,q,base='jaka_base_link',end='tool0',old=False):
    """Independent vectorized URDF chain; fixed RPY + revolute axis Rodrigues."""
    tree=ET.fromstring(xml);joints={x.find('child').get('link'):x for x in tree.findall('joint')}
    chain=[];link=end
    while link!=base:
        joint=joints[link];chain.append(joint);link=joint.find('parent').get('link')
    r=np.broadcast_to(np.eye(3),(len(q),3,3)).copy();p=np.zeros((len(q),3));origins=[];axes=[]
    for joint in reversed(chain):
        origin=joint.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        angles=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        if old and joint.get('name')=='tool0_joint':angles=np.zeros(3)
        p+=np.einsum('nij,j->ni',r,xyz);r=r@rpy(angles)
        if joint.get('type') in ('revolute','continuous'):
            index=int(joint.get('name').split('_')[-1])-1;axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ');axis/=np.linalg.norm(axis)
            origins.append(p.copy());axes.append(np.einsum('nij,j->ni',r,axis))
            skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0.]])
            a=q[:,index];rot=np.eye(3)+np.sin(a)[:,None,None]*skew+(1-np.cos(a))[:,None,None]*(skew@skew);r=r@rot
    jac=np.stack([np.r_[np.cross(ax,p-pos).T,ax.T].T for ax,pos in zip(axes,origins)],axis=2)
    return p,r,jac


def alignment(a,b,mask):
    valid=mask & np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)&(np.linalg.norm(a,axis=1)>1e-10)&(np.linalg.norm(b,axis=1)>1e-10)
    c=np.sum(a[valid]*b[valid],axis=1)/(np.linalg.norm(a[valid],axis=1)*np.linalg.norm(b[valid],axis=1))
    return {'count':len(c),'positive_count':int(np.sum(c>0)),'positive_fraction':None if not len(c) else float(np.mean(c>0)), 'median_cosine':None if not len(c) else float(np.median(c))}


def recorded_urdf(metadata):
    # Snapshot keys contain the original host's absolute checkout path.
    matches=[value for path,value in metadata['sources'].items()
             if path.endswith('/'+URDF_RELATIVE)]
    if len(matches)!=1:
        raise ValueError('expected exactly one recorded JAKA URDF snapshot')
    return matches[0]


def audit(directory,out):
    m=json.loads((directory/'metadata.json').read_text());rows=[json.loads(l) for l in (directory/'samples.jsonl').open()]
    snapshot=recorded_urdf(m);xml=snapshot['content']
    q=np.array([row['measured_q_rad'] for row in rows]);p,r,jac=fk(xml,q);po,ro,_=fk(xml,q,old=True)
    n=len(rows);t=np.array([row['elapsed_s'] for row in rows]);active=np.array([bool(row.get('admittance_updated')) and not row.get('fault') and bool(row.get('command_sent')) for row in rows])
    f=np.full((n,3),np.nan);v=np.full((n,3),np.nan);x=np.full((n,3),np.nan);ext_error=[];application=[];dynamics=[];rotation_error=[];p_logged=np.array([row['measured_tcp_pose']['position_m'] for row in rows]);rc=np.array(m['sensor_to_tool_rotation']);d=np.diag([-1.,-1.,1.])
    for i,row in enumerate(rows):
        rl=rotation_matrix(np.array(row['measured_tcp_pose']['quaternion_wxyz']));rotation_error.append(np.max(np.abs(r[i]-rl)))
        if not row.get('admittance_updated'):continue
        a,b=row['admittance_integrated'],row['admittance_before'];w=row['wrench_stages'];f[i]=row['admittance_input_wrench'][:3];v[i]=a['velocity'][:3];x[i]=a['offset'][:3]
        ext_error.append(np.array(w['external_tcp_unfiltered'][:3])-rc@np.array(w['compensated_sensor'][:3]))
        application.append(np.array(row['unclamped_pose']['position_m'])-np.array(a['nominal_pose']['position_m'])-r[i]@x[i])
        mass=np.array(m['effective_args']['adm_mass']);damp=np.array(m['effective_args']['adm_damping']);stiff=np.array(m['effective_args']['adm_stiffness']);limit=np.array(m['effective_args']['adm_max_velocity']);dt=row['dt_s'];v0=np.array(b['velocity']);x0=np.array(b['offset']);force=np.array(row['admittance_input_wrench']);v1=np.clip(v0+dt*(force-damp*v0-stiff*x0)/mass,-limit,limit);dynamics.append(np.r_[v1-np.array(a['velocity']),x0+dt*v1-np.array(a['offset'])])
    fb=np.einsum('nij,nj->ni',r,f);vb=np.einsum('nij,nj->ni',r,v);candidate=np.einsum('nij,nj->ni',r,np.einsum('ij,nj->ni',d,f))
    mv=np.full((n,3),np.nan);win_mask=np.zeros(n,bool)
    for i in range(n-5):
        if active[i:i+6].all() and .025<t[i+5]-t[i]<.06:
            mv[i]=(p[i+5]-p[i])/(t[i+5]-t[i]);win_mask[i]=True
    force_mask=win_mask&(np.linalg.norm(fb,axis=1)>1.5)&(np.linalg.norm(mv,axis=1)>.003)
    velocity_mask=win_mask&(np.linalg.norm(vb,axis=1)>.003)&(np.linalg.norm(mv,axis=1)>.003)
    sustained=force_mask&(np.linalg.norm(vb,axis=1)>.003)
    for i in np.flatnonzero(sustained):
        norms=np.linalg.norm(fb[i:i+6],axis=1);dots=np.sum(fb[i:i+6]*fb[i],axis=1)/np.maximum(norms*np.linalg.norm(fb[i]),1e-30)
        sustained[i]=bool(np.min(norms)>1.5 and np.min(dots)>.95)
    # Compare actual motion with commanded FK increments, with a 40ms lag.
    commands=np.array([row.get('final_command_q_rad',row['measured_q_rad']) for row in rows]);pc,_,_=fk(xml,commands);cv=np.full_like(p,np.nan);cv[1:]=(pc[1:]-pc[:-1])/np.diff(t)[:,None]
    result={'session':directory.name,'rows':n,'active_cycles':int(active.sum()),'fault_counts':{str(name):sum(row.get('fault')==name for row in rows) for name in set(row.get('fault') for row in rows)},'keys_pressed_cycles':sum(bool(row.get('pressed_before')) for row in rows),
        'independent_fk_max_position_error_m':float(np.max(np.abs(p-p_logged))),'independent_fk_max_rotation_matrix_error':float(np.max(rotation_error)),
        'tool0_flip_max_position_change_m':float(np.max(np.abs(p-po))),'tool0_new_equals_old_times_D_error':float(np.max(np.abs(r-ro@d))),
        'external_rotation_max_error_N':float(np.max(np.abs(ext_error))),'admittance_application_max_error_m':float(np.max(np.abs(application))),'discrete_replay_max_error':float(np.max(np.abs(dynamics))),
        'measured_vs_logged_force':alignment(fb,mv,force_mask),'measured_vs_logged_force_sustained':alignment(fb,mv,sustained),'measured_vs_admittance_velocity':alignment(vb,mv,velocity_mask),
        'measured_vs_frame_migrated_force_sustained':alignment(candidate,mv,sustained),'command_increment_vs_admittance_velocity':alignment(cv,vb,active&(np.linalg.norm(cv,axis=1)>.003)&(np.linalg.norm(vb,axis=1)>.003)),
        'current_and_logged_urdf_identical':hashlib.sha256((ROOT/URDF_RELATIVE).read_bytes()).hexdigest()==snapshot['sha256'],
        'initial_tool_axes_in_base':r[0].tolist(),'thresholds':{'force_N':1.5,'speed_m_s':.003,'forward_window_samples':5,'direction_sustained_min_cosine':.95},
        'sensor_to_tool_rotation_logged':rc.tolist(),'frame_migrated_rotation_D_times_logged':(d@rc).tolist(),'tool_to_sensor_m_logged':m['tool_to_sensor_m'],'frame_migrated_tool_to_sensor_m':(d@np.array(m['tool_to_sensor_m'])).tolist()}
    with (out/(directory.name+'_samples.csv')).open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['sequence','time_s','active','included_sustained',*[prefix+'_'+axis for prefix in ['fk_position_m','measured_velocity_m_s','admittance_velocity_m_s','logged_force_N','candidate_frame_force_N'] for axis in 'xyz']])
        for i,row in enumerate(rows):writer.writerow([row['sequence'],t[i],bool(active[i]),bool(sustained[i]),*p[i],*mv[i],*vb[i],*fb[i],*candidate[i]])
    np.savez(out/(directory.name+'_arrays.npz'),t=t,p=p,mv=mv,vb=vb,fb=fb,candidate=candidate,active=active,mask=sustained)
    return result,xml,q,p,r,jac


def main():
    out=ROOT/'logs/force_direction_audit';out.mkdir(exist_ok=True);results=[]
    for name in ('jaka_trial_001','jaka_trial_002','jaka_trial_003'):
        directory=ROOT/'logs'/name
        result,xml,q,p,r,jac=audit(directory,out)
        # Use installed Pinocchio in system NumPy environment as a separate cross-check.
        from replace_disk_robot.kinematics.jaka import JakaKinematics
        from replace_disk_robot.core import JointState
        path=Path('/tmp/jaka_audit_recorded.urdf');path.write_text(xml)
        kin=JakaKinematics(path);pe=[];re=[];je=[]
        for i in np.unique(np.linspace(0,len(q)-1,30,dtype=int)):
            state=JointState(kin.joint_names,q[i]);pose=kin.forward(state);pe.append(np.max(np.abs(pose.position_m-p[i])));re.append(np.max(np.abs(rotation_matrix(pose.quaternion_wxyz)-r[i])));je.append(np.max(np.abs(kin.jacobian(state)-jac[i])))
        result['pinocchio_30_poses_max_position_error_m']=float(np.max(pe));result['pinocchio_30_poses_max_rotation_error']=float(np.max(re));result['pinocchio_30_poses_max_jacobian_error']=float(np.max(je))
        assert result['independent_fk_max_position_error_m'] < 1e-10
        assert result['pinocchio_30_poses_max_jacobian_error'] < 1e-9
        results.append(result);print(json.dumps(result,indent=2))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    # Offline refit of existing calibration CSV, using corrected frame migration.
    from identify_ft_payload import StaticPose,identify_payload,load_all_rows
    poses,transforms=load_all_rows(ROOT/'tool/ft_gravity_samples.csv',20)
    with (ROOT/'tool/ft_gravity_samples.csv').open() as file:csvrows=list(csv.DictReader(file))
    q=np.array([[float(row[f'q{i}_rad']) for i in range(1,7)] for row in csvrows]);p,r,jac=fk(xml,q)
    old_r=transforms['rotation_sensor_to_tool'];new_r=np.diag([-1.,-1.,1.])@old_r
    new_poses=[StaticPose(a.capture_id,a.samples,r[i]@new_r,a.force_sensor_n,a.torque_sensor_nm) for i,a in enumerate(poses)]
    fit=identify_payload(new_poses, gravity_mode='free')
    summary={k:fit[k] for k in ['mass_kg','signed_gravity_force_base_n','gravity_tilt_deg','center_of_mass_sensor_m','force_bias_sensor_n','torque_bias_sensor_nm']};summary['fit']={k:v for k,v in fit['fit'].items() if not k.endswith('weights')}
    summary['csv_old_orientation_matches_current_fk_times_old_extrinsic_error']=float(np.max([np.max(np.abs(a.rotation_base_sensor-r[i]@old_r)) for i,a in enumerate(poses)]))
    summary['calibration_command_active_rows']=sum(int(row['command_active'])!=0 for row in csvrows)
    (out/'candidate_calibration_refit_summary.json').write_text(json.dumps(summary,indent=2));print('REFIT',json.dumps(summary,indent=2))

if __name__=='__main__':main()
