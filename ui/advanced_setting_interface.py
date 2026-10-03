"""
@desc: 高级设置页面
"""
import sys

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout,
                           FluentIcon, NavigationWidget, NavigationItemPosition,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           HyperlinkCard, PrimaryPushSettingCard, ComboBoxSettingCard, PushSettingCard,
                           ExpandGroupSettingCard, ComboBox, isDarkTheme,
                           MessageBox, qconfig)
from backend.config import (config, tr, tr_fallback, VERSION, PROJECT_HOME_URL,
                            PROJECT_ISSUES_URL, PROJECT_RELEASES_URL)
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor
from backend.tools.constant import VideoSubFinderDecoder

class HelpIconButton(QtWidgets.QAbstractButton):
    """
    问号帮助按钮, 用来替换 ExpandSettingCard 自带的旋转箭头。

    刻意不继承 qfluentwidgets 的按钮: 它们的 __init__ 被一个自定义 overload 装饰器包着,
    装饰器内部会调 self.__init__(...), 到了子类身上这个 self.__init__ 又指回子类, 于是
    无限递归。这里只实现 HeaderSettingCard 会调的三个方法。
    """

    BUTTON_SIZE = 30
    ICON_SIZE = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
        self.setCheckable(True)
        self.isHover = False
        self.isPressed = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing
                               | QtGui.QPainter.SmoothPixmapTransform)
        r = 255 if isDarkTheme() else 0

        if not self.isEnabled():
            painter.setOpacity(0.36)
            color = QtCore.Qt.transparent
        elif self.isPressed:
            color = QtGui.QColor(r, r, r, 10)
        elif self.isHover:
            color = QtGui.QColor(r, r, r, 14)
        elif self.isChecked():
            color = QtGui.QColor(r, r, r, 10)
        else:
            color = QtCore.Qt.transparent

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(self.rect(), 4, 4)

        # 用和「开发配置」页一样的问号字形, 而不是文字 "?"
        offset = (self.BUTTON_SIZE - self.ICON_SIZE) / 2
        FluentIcon.QUESTION.render(
            painter, QtCore.QRectF(offset, offset, self.ICON_SIZE, self.ICON_SIZE))

    def enterEvent(self, event):
        self.setHover(True)

    def leaveEvent(self, event):
        self.setHover(False)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.setPressed(True)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setPressed(False)

    def setHover(self, isHover):
        self.isHover = isHover
        self.update()

    def setPressed(self, isPressed):
        self.isPressed = isPressed
        self.update()

    def setExpand(self, isExpand):
        """ExpandGroupSettingCard 通过箭头的接口调到这里, 问号只需同步选中态。"""
        self.setChecked(isExpand)


class ModelChoiceCard(ExpandGroupSettingCard):
    """
    可折叠的选择卡片: 头部是问号和下拉框, 主体是该设置的逐选项说明。

    折叠主体的高度由 ExpandGroupSettingCard._adjustViewSize 按子控件的 sizeHint() 求和
    得出, 而开了 wordWrap 的 QLabel 在 sizeHint 里只报一行, 文字会被裁掉。所以说明文字
    自己带换行(在 ini 里写成续行), 也不要改成 setWordWrap(True)。
    """

    def __init__(self, configItem, icon, title, content, texts, helpText, parent=None):
        """
        texts 会和 configItem.options 逐个 zip, 所以两者的顺序和长度必须一致。helpText
        需要自带换行, 原因见类 docstring。
        """
        super().__init__(icon, title, content, parent)
        self.configItem = configItem

        self._replace_expand_button()

        self.comboBox = ComboBox(self)
        self.optionToText = {option: text for option, text in zip(configItem.options, texts)}
        for text, option in zip(texts, configItem.options):
            self.comboBox.addItem(text, userData=option)
        self.comboBox.setCurrentText(self.optionToText[configItem.value])
        self.comboBox.currentIndexChanged.connect(self._onCurrentIndexChanged)
        configItem.valueChanged.connect(self.setValue)
        self.card.addWidget(self.comboBox)

        body = QtWidgets.QWidget(self.view)
        bodyLayout = QtWidgets.QVBoxLayout(body)
        bodyLayout.setContentsMargins(20, 12, 20, 16)
        self.helpLabel = QtWidgets.QLabel(helpText, body)
        self.helpLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bodyLayout.addWidget(self.helpLabel)
        self.addGroupWidget(body)

    def _replace_expand_button(self):
        """
        把默认的旋转箭头换成问号, 并接管卡片上的事件过滤。

        库的 HeaderSettingCard.eventFilter 在整张卡片收到 QEvent.Enter 时就点亮箭头, 那样
        鼠标停在卡片任意位置问号都像被悬停。这里只处理按下和松开, 悬停交给按钮自己。
        """
        old = self.card.expandButton
        index = self.card.hBoxLayout.indexOf(old)
        self.helpButton = HelpIconButton(self.card)
        self.card.hBoxLayout.insertWidget(index, self.helpButton, 0, QtCore.Qt.AlignRight)
        self.card.hBoxLayout.removeWidget(old)
        # 只 removeWidget 不够: 旧按钮仍挂在父控件上且可见, 会一直画在 (0, 0)
        old.setParent(None)
        old.deleteLater()
        self.card.expandButton = self.helpButton
        self.helpButton.clicked.connect(self.toggleExpand)

        self.card.removeEventFilter(self.card)
        self.card.installEventFilter(self)

    def eventFilter(self, obj, event):
        """整张卡片依然可以点击展开, 但不再让卡片去点亮图标。"""
        if obj is self.card:
            if event.type() == QtCore.QEvent.MouseButtonPress \
                    and event.button() == QtCore.Qt.LeftButton:
                self.helpButton.setPressed(True)
            elif event.type() == QtCore.QEvent.MouseButtonRelease \
                    and event.button() == QtCore.Qt.LeftButton:
                self.helpButton.setPressed(False)
                self.helpButton.click()
        return super().eventFilter(obj, event)

    def _onCurrentIndexChanged(self, index: int):
        qconfig.set(self.configItem, self.comboBox.itemData(index))

    def setValue(self, value):
        if value not in self.optionToText:
            return
        self.comboBox.setCurrentText(self.optionToText[value])
        qconfig.set(self.configItem, value)


class AdvancedSettingInterface(ScrollArea):
    """高级设置页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__initWidget()

    def __initWidget(self):
        # 创建滚动内容的容器
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)
        
        # 设置滚动区域属性
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # 设置滚动区域样式以适应主题
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        
        # 设置UI
        self.setup_ui()
        self.setup_layout()

    def setup_layout(self):
        self.advanced_group.addSettingCard(self.det_model)
        self.advanced_group.addSettingCard(self.rec_model)
        self.advanced_group.addSettingCard(self.frame_extraction)
        self.advanced_group.addSettingCard(self.rec_batch_number)
        self.advanced_group.addSettingCard(self.max_batch_size)
        self.advanced_group.addSettingCard(self.subtitle_area)
        self.advanced_group.addSettingCard(self.extract_frequency)
        self.advanced_group.addSettingCard(self.tolerant_pixel_y)
        self.advanced_group.addSettingCard(self.tolerant_pixel_x)
        self.advanced_group.addSettingCard(self.subtitle_area_deviation_pixel)
        self.advanced_group.addSettingCard(self.waterark_area_num)
        self.advanced_group.addSettingCard(self.threshold_text_similarity)
        self.advanced_group.addSettingCard(self.drop_score)
        self.advanced_group.addSettingCard(self.subtitle_area_deviation_rate)
        self.advanced_group.addSettingCard(self.save_directory)
        self.advanced_group.addSettingCard(self.check_update_on_startup)
        self.expandLayout.addWidget(self.advanced_group)

        self.video_sub_finder_group.addSettingCard(self.video_sub_finder_cpu_cores)
        self.video_sub_finder_group.addSettingCard(self.video_sub_finder_decoder)
        self.expandLayout.addWidget(self.video_sub_finder_group)

        self.dev_group.addSettingCard(self.debug_ocr_loss)
        self.dev_group.addSettingCard(self.debug_no_delete_cache)
        self.dev_group.addSettingCard(self.delete_empty_time_stamp)
        self.expandLayout.addWidget(self.dev_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        self.expandLayout.addWidget(self.about_group)
       
        self.expandLayout.setSpacing(16)
        self.expandLayout.setContentsMargins(16, 16, 16, 48)
        
    def setup_ui(self):
        """设置UI"""
        # 高级设置组
        self.advanced_group = SettingCardGroup(tr["Setting"]["AdvancedSetting"], self.scrollWidget)
        # VideoSubFinder设置组
        self.video_sub_finder_group = SettingCardGroup(tr["Setting"]["VideoSubFinderSetting"], self.scrollWidget)
        # 开发设置组  
        self.dev_group = SettingCardGroup(tr["Setting"]["DevSetting"], self.scrollWidget)
        # 关于设置组  
        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        
        self.det_model = ModelChoiceCard(
            configItem=config.detModel,
            icon=FluentIcon.VIEW,
            title=tr["Setting"]["DetModel"],
            content=tr["Setting"]["DetModelDesc"],
            parent=self.advanced_group,
            texts=list(config.detModel.validator.options),
            helpText=tr_fallback("ModelHelp", "DetModel"),
        )
        self.rec_model = ModelChoiceCard(
            configItem=config.recModel,
            icon=FluentIcon.FONT,
            title=tr["Setting"]["RecModel"],
            content=tr["Setting"]["RecModelDesc"],
            parent=self.advanced_group,
            texts=list(config.recModel.validator.options),
            helpText=tr_fallback("ModelHelp", "RecModel"),
        )
        self.frame_extraction = ModelChoiceCard(
            configItem=config.frameExtraction,
            icon=FluentIcon.SPEED_HIGH,
            title=tr["Setting"]["FrameExtraction"],
            content=tr["Setting"]["FrameExtractionDesc"],
            parent=self.advanced_group,
            texts=[tr_fallback('FrameExtraction', i)
                   for i in config.frameExtraction.validator.options],
            helpText=tr_fallback("ModelHelp", "FrameExtraction"),
        )

        # 每张图中同时识别的文本框数量
        self.rec_batch_number = RangeSettingCard(
            configItem=config.recBatchNumber,
            icon=FluentIcon.SEARCH,
            title=tr["Setting"]["RecBatchNumber"],
            content=tr["Setting"]["RecBatchNumberDesc"],
            parent=self.advanced_group
        )
        # DB算法每个batch识别多少张
        self.max_batch_size = RangeSettingCard(
            configItem=config.maxBatchSize,
            icon=FluentIcon.SEARCH_MIRROR,
            title=tr["Setting"]["MaxBatchSize"],
            content=tr["Setting"]["MaxBatchSizeDesc"],
            parent=self.advanced_group
        )
        # 字幕出现区域
        self.subtitle_area = ComboBoxSettingCard(
            configItem=config.subtitleArea,
            icon=FluentIcon.VIEW,
            title=tr["Setting"]["SubtitleArea"],
            content=tr["Setting"]["SubtitleAreaDesc"],
            parent=self.advanced_group,
            texts=tr['SubtitleArea'].values(),
        )
        # 每一秒抓取多少帧进行OCR识别
        self.extract_frequency = RangeSettingCard(
            configItem=config.extractFrequency,
            icon=FluentIcon.SPEED_HIGH,
            title=tr["Setting"]["ExtractFrequency"],
            content=tr["Setting"]["ExtractFrequencyDesc"],
            parent=self.advanced_group
        )
        # 容忍的像素点偏差
        self.tolerant_pixel_y = RangeSettingCard(
            configItem=config.tolerantPixelY,
            icon=FluentIcon.ARROW_DOWN,
            title=tr["Setting"]["TolerantPixelY"],
            content=tr["Setting"]["TolerantPixelYDesc"],
            parent=self.advanced_group
        )
        self.tolerant_pixel_x = RangeSettingCard(
            configItem=config.tolerantPixelX,
            icon=FluentIcon.RIGHT_ARROW,
            title=tr["Setting"]["TolerantPixelX"],
            content=tr["Setting"]["TolerantPixelXDesc"],
            parent=self.advanced_group
        )
        # 字幕区域偏移量
        self.subtitle_area_deviation_pixel = RangeSettingCard(
            configItem=config.subtitleAreaDeviationPixel,
            icon=FluentIcon.MOVE,
            title=tr["Setting"]["SubtitleAreaDeviationPixel"],
            content=tr["Setting"]["SubtitleAreaDeviationPixelDesc"],
            parent=self.advanced_group
        )
        # 最有可能出现的水印区域
        self.waterark_area_num = RangeSettingCard(
            configItem=config.waterarkAreaNum,
            icon=FluentIcon.PHOTO,
            title=tr["Setting"]["WaterarkAreaNum"],
            content=tr["Setting"]["WaterarkAreaNumDesc"],
            parent=self.advanced_group
        )
        # 文本相似度阈值
        self.threshold_text_similarity = RangeSettingCard(
            configItem=config.thresholdTextSimilarity,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["ThresholdTextSimilarity"],
            content=tr["Setting"]["ThresholdTextSimilarityDesc"],
            parent=self.advanced_group
        )
        # 字幕提取中置信度低于0.75的不要
        self.drop_score = RangeSettingCard(
            configItem=config.dropScore,
            icon=FluentIcon.ACCEPT_MEDIUM,
            title=tr["Setting"]["DropScore"],
            content=tr["Setting"]["DropScoreDesc"],
            parent=self.advanced_group
        )
        # 字幕区域允许偏差
        self.subtitle_area_deviation_rate = RangeSettingCard(
            configItem=config.subtitleAreaDeviationRate,
            icon=FluentIcon.PIE_SINGLE,
            title=tr["Setting"]["SubtitleAreaDeviationRate"],
            content=tr["Setting"]["SubtitleAreaDeviationRateDesc"],
            parent=self.advanced_group
        )
        # 视频保存路径
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.advanced_group
        )
        self.save_directory.clicked.connect(self.choose_save_directory)
        # 启动时检查应用更新
        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.advanced_group
        )
        # VideoSubFinder CPU核心数
        self.video_sub_finder_cpu_cores = RangeSettingCard(
            configItem=config.videoSubFinderCpuCores,
            icon=FluentIcon.SPEED_MEDIUM,
            title=tr["Setting"]["VideoSubFinderCpuCores"],
            content=tr["Setting"]["VideoSubFinderCpuCoresDesc"],
            parent=self.video_sub_finder_group
        )
        # VideoSubFinder 视频解码组件
        self.video_sub_finder_decoder = ComboBoxSettingCard(
            configItem=config.videoSubFinderDecoder,
            icon=FluentIcon.VIDEO,
            title=tr["Setting"]["VideoSubFinderDecoder"],
            content=tr["Setting"]["VideoSubFinderDecoderDesc"],
            texts=[item.value for item in VideoSubFinderDecoder],
            parent=self.video_sub_finder_group
        )
        # 输出丢失的字幕帧
        self.debug_ocr_loss = SwitchSettingCard(
            configItem=config.debugOcrLoss,
            icon=FluentIcon.QUESTION,
            title=tr["Setting"]["DebugOcrLoss"],
            content=tr["Setting"]["DebugOcrLossDesc"],
            parent=self.dev_group
        )
        # 是否不删除缓存数据
        self.debug_no_delete_cache = SwitchSettingCard(
            configItem=config.debugNoDeleteCache,
            icon=FluentIcon.FOLDER,
            title=tr["Setting"]["DebugNoDeleteCache"],
            content=tr["Setting"]["DebugNoDeleteCacheDesc"],
            parent=self.dev_group
        )
        # 是否删除空时间轴
        self.delete_empty_time_stamp = SwitchSettingCard(
            configItem=config.deleteEmptyTimeStamp,
            icon=FluentIcon.DELETE,
            title=tr["Setting"]["DeleteEmptyTimeStamp"],
            content=tr["Setting"]["DeleteEmptyTimeStampDesc"],
            parent=self.dev_group
        )
        # 添加反馈链接
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(PROJECT_ISSUES_URL)
        ))
        # 添加版权信息
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())
        # 添加项目链接
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """ show message box """
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """ check software update

        Parameters
        ----------
        ignore: bool
            ignore message box when no updates are available
        """
        TaskExecutor.runTask(self.version_manager.has_new_version).then(
            lambda success: self.on_version_info_fetched(success, ignore))

    def on_version_info_fetched(self, success, ignore=False):
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True,
                lambda: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(PROJECT_RELEASES_URL)
                )
            )
        elif not ignore:
            self.show_message_box(
                tr["Setting"]["NoUpdatesAvailableTitle"],
                tr["Setting"]["NoUpdatesAvailableDesc"],
            )
    
    def choose_save_directory(self):
        """选择保存目录"""
        last_save_directory = "./" if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], last_save_directory)
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)
            
    def macos_scrollarea_issue_workaround(self):
        if sys.platform != "darwin":
            return
        self.verticalScrollBar().setValue(0)
        self.scrollWidget.adjustSize()
        self.expandLayout.update()
        self.expandLayout.activate()
        
    def resizeEvent(self, event):
        # macos_scrollarea_issue_workaround
        if sys.platform == "darwin":
            self.verticalScrollBar().setValue(0)
        super().resizeEvent(event)