# �|�P�������ނɈ��ʐ��_�����H���Ă݂�
# https://tepppei.hatenablog.com/

## �����X�^�[�{�[���ƃX�[�p�[�{�[���̕߂܂��₷���ɗL�Ӎ����邩���� ##
# csv�� https://github.com/Teppei-Kanayama/pokemon_get_simulator/blob/master/resources/data.csv
df <- read.table("pokemon_ball_data.csv",
                header=T, 
                sep=",",
                fileEncoding="UTF-8")

# �T���v�����O���ꂽ�����X�^�[�{�[���ƃX�[�p�[�{�[����������
m_throwns <- df[df$ball_type=='monster', ]$thrown_balls
s_throwns <- df[df$ball_type=='super', ]$thrown_balls
print(paste("length(m_throwns):", length(m_throwns)))
print(paste("length(s_throwns):", length(s_throwns)))

# ���ϒl,�s�ϕ��U�m�F
print(paste("m_throwns mean var:", mean(m_throwns), var(m_throwns)))
print(paste("s_throwns mean var:", mean(s_throwns), var(s_throwns)))

# 2�Q�������U��F���� �A�������i2�Q�̕��U�ɍ����Ȃ��j
var.test(x=m_throwns, y=s_throwns, conf.level=0.95)  
# p�l<0.05�Ȃ̂ŁA�A���������p=2�Q�̕��U�ɍ�������

# �Ή��̂Ȃ��A�E�F���`��t����i�s�����U��t����j �A�������i2�Q�̕��ςɍ����Ȃ��j
t.test(x=m_throwns, y=s_throwns, conf.level=0.95, var.equal=F, paired=F)
# p�l>0.05�Ȃ̂ŁA�A���������p�ł��Ȃ�=���ς̍����Ȃ�
# �X�[�p�[�{�[���̕������\�����Ƃ͂Ȃ�Ȃ�����
# ���ςƈႤ�̂ŃZ���N�V�����o�C�A�X�̉\������


## �𗍈��q�̑��� ##
# �T���v���𕪐͂������ʁA
# �u�߂܂��₷�����ȃ|�P�����ɂ̓����X�^�[�{�[�����g���A�߂܂��ɂ������ȃ|�P�����ɂ̓X�[�p�[�{�[�����g���Ă���v
# �܂�A�����_���T���v�����O�ł͂Ȃ����Ƃ����肵���B
# ��قǂ�t����̓Z���N�V�����o�C�A�X�ɂ���Ęc��ł���\�����o�Ă����B

# �T���v���𕪐͂������ʂ���A
# �u�ǂ̃{�[�����g�����v�i�ړI�ϐ��j�́u�߂܂���̂Ɏg�����{�[���̌��v�i�����ϐ��j�ɂ݈̂ˑ�����O�񂾂������A
# �u�|�P�����̂��܂��₷���v�̕ϐ�����L2�ϐ��̗����ɑ΂��đ��ւ����\�����������Ƃ��킩��B
# ���̂悤�� �ړI�ϐ��Ɛ����ϐ��ǂ���ɂ����ւ�����ϐ����𗍈��q�ƌĂсA
# �𗍈��q�𖳎����Č��ʌ��؂��s���ƁA�Z���N�V�����o�C�A�X�ɂ����ۂ̌��ʂƈقȂ錋�ʂ��o�Ă��܂��B

# �u�|�P�����̂��܂��₷���v�̉e������菜�����߂ɐ��`�d��A��p�������f�������s���������ŁA
# ���߂āu�X�[�p�[�{�[���͖{���Ƀ����X�^�[�{�[�����߂܂��₷���̂��H�v�Ƃ������������؂���
# ���������python�ŉ�A���f������ 



