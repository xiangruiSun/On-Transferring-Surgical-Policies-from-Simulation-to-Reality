"""2026-11-10 evening (recalibration after the restart; copy of fit_roll_1110.py, data/tips/output paths only). ORIGINAL NOTE: 2026-11-10 (new setup, rolls -50..+90; copy of fit_roll_1109.py, data/tips/output paths only). ORIGINAL NOTE: 2026-11-09 14:3x (recalibration in the new scene; copy of fit_roll_1108.py, data/tips/output paths only): refit the 11-04 roll model (A_check/fit_roll.py method) on today's roll_data with today's calibration. Writes roll_model_1110pm3.json only."""
import json,sys,numpy as np,cv2
R0='~/rl_live/sim2real_20261106/S1/';sys.path.insert(0,R0+'code')
import autotpl as AT
np.set_printoptions(precision=2,suppress=True)
FX,CX,CY=1860.724932,690.538345,576.803638;FB=FX*5.4476353950418295;proj=lambda X:np.array([FX*X[0]/X[2]+CX,FX*X[1]/X[2]+CY])
TCP=np.array([-2.70,1.08,23.81]);cal=json.load(open(R0+'C_r6cal/calibration_tcp.json'));M=np.array(cal['M']);A3=M[:3];inv=np.linalg.inv(A3);Rref=np.array(cal['base_T'])[:3,:3]
hand={-50:[(742,212),(768,427)],-25:[(779,162),(732,392)],0:[(829,129),(708,333)],25:[(885,125),(708,279)]}   # 11-10 19:1x tips read by eye on roll_data_1110pm3b right_rect (gridded 1.2x views, ~5-10 px; collected 10 mm below the base in the image so both blades stay in view)
D=[]
for r,tips in hand.items():
    d=R0+f'A_check/roll_data_1110pm3b/roll{r:+d}/';st=json.load(open(d+'state.json'));T=np.array(st['sp']);R=T[:3,:3];cp=T[:3,3]*1e3;Xt=np.r_[cp+R@TCP,1]@M
    Rg=cv2.imread(d+'right_rect.png',0).astype(np.float32);Lg=cv2.imread(d+'left_rect.png',0).astype(np.float32)
    s_=[AT._ncc_disp(Rg,Lg,u,v,FB/Xt[2],span=30,h=12) for u,v in tips]
    Xs=[np.array([(u-CX)*(FB/s[1])/FX,(v-CY)*(FB/s[1])/FX,FB/s[1]]) for (u,v),s in zip(tips,s_)];Xm=np.mean(Xs,0);off=R.T@((Xm-Xt)@inv)
    span=np.linalg.norm(Xs[0]-Xs[1]);D.append(dict(r=r,R=R,off=off,ncc=[round(s[0],2) for s in s_],span=span,Xt=Xt,tips=tips))
    print(f'roll {r:+4d} q4 {np.degrees(st["js"][3]):5.0f} deg offset {off} span {span:.1f} mm ncc {D[-1]["ncc"]} disp {[round(s[1],1) for s in s_]}')
G=[x for x in D if min(x['ncc'])>=0.4 and 8.5<=x['span']<=11.0];print('fit set',[x['r'] for x in G],'excluded',[(x['r'],x['ncc'],round(x['span'],1)) for x in D if x not in G])
design=lambda R:np.c_[np.eye(3),-(R.T@Rref)[:,:2]]
fit=lambda S:np.linalg.lstsq(np.vstack([design(x['R']) for x in S]),np.concatenate([x['off'] for x in S]),rcond=None)[0]
px=lambda x,o:float(np.linalg.norm(proj(x['Xt']+(x['R']@o)@A3)-np.mean(x['tips'],0)))
OLD=np.r_[5.19,0.51,-7.15,1.59,1.32]   # 1108 constants now in attempt.py
p=fit(G);print('fit a',p[:3],'e_xy',p[3:],'|e_xy| %.2f'%np.linalg.norm(p[3:]))
print('LOO image px of jaw midpoint: new model | 11-04 model (19 px ~ 1 mm at 11-04 range; today ~16 px/mm)')
E=[]
for x in G:
    pl=fit([y for y in G if y is not x]);a,b=px(x,design(x['R'])@pl),px(x,design(x['R'])@OLD);E.append((a,b));print(f'  roll {x["r"]:+4d}: {a:4.0f} | {b:4.0f} px')
E=np.array(E);print('LOO RMS %.0f | %.0f px, max %.0f | %.0f'%(*np.sqrt((E**2).mean(0)),*E.max(0)))
print('in-sample px:',{x['r']:round(px(x,design(x['R'])@p)) for x in G})
json.dump(dict(a=p[:3].tolist(),e_xy=p[3:].tolist(),Rref=Rref.tolist(),fit_set=[x['r'] for x in G],tips=hand,calibration=R0+'C_r6cal/calibration_tcp.json'),open(R0+'A_check/roll_model_1110pm3.json','w'),indent=1)
